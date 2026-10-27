#include "vc_object_store/objects/AttachedDocument.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <cmath>

namespace vc::objects {
namespace {
geometry::Point position(const nlohmann::json& value)
{
    if (value.is_array() && value.size() >= 3 && value[0].is_number() && value[1].is_number() && value[2].is_number()) {
        const geometry::Point xyz{value[0].get<double>(), value[1].get<double>(), value[2].get<double>()};
        if (std::isfinite(xyz[0]) && std::isfinite(xyz[1]) && std::isfinite(xyz[2])) return xyz;
    }
    return {-1, -1, -1};
}
}

void attachDocument(ObjectStore& store, WriteTxn& txn, const std::string& objectUuid, const std::string& collectionUuid,
                    const nlohmann::json& document, const DocumentLayout& layout, const std::string& name,
                    const nlohmann::json& coordinates)
{
    auto records = nlohmann::json::array();
    for (const auto& record : document.value(layout.listKey, nlohmann::json::array()))
        if (record.is_object()) records.push_back(record);
    geometry::PointCollection points;
    for (const auto& record : records) {
        points.ids.push_back(geometry::PointId(points.ids.size()));
        points.points.push_back(position(record.value(layout.positionKey, nlohmann::json())));
    }
    auto rest = document;
    rest.erase(layout.listKey);
    ObjectRecord record;
    record.uuid = collectionUuid;
    record.kind = "point_collection";
    record.name = name;
    record.meta = nlohmann::json{{"role", layout.role},
                                 {"coordinates", coordinates},
                                 {"source", {{"document", rest},
                                             {"list_key", layout.listKey},
                                             {"position_key", layout.positionKey},
                                             {"point_attributes", records}}}}.dump();
    store.upsertObject(txn, record);
    store.putGeometry(txn, collectionUuid, geometry::encode(points));
    store.attach(txn, collectionUuid, objectUuid, layout.role);
}

std::optional<nlohmann::json> attachedDocument(const ObjectStore& store, const Txn& txn, const std::string& objectUuid,
                                               const std::string& role)
{
    for (const auto& edge : store.attachments(txn, objectUuid, role)) {
        const auto uuid = edge.aUuid == objectUuid ? edge.bUuid : edge.aUuid;
        const auto object = store.objectByUuid(txn, uuid);
        const auto head = object && !object->retired ? store.currentGeometry(txn, uuid) : std::nullopt;
        if (!head) continue;
        const auto source = nlohmann::json::parse(object->meta).value("source", nlohmann::json::object());
        auto document = source.value("document", nlohmann::json::object());
        document[source.value("list_key", std::string("records"))] =
            nlohmann::json::parse(store.geometryPointAnnotations(txn, head->uuid, "points"));
        return document;
    }
    return std::nullopt;
}

std::optional<nlohmann::json> attachedDocument(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                                               const std::string& objectUuid, const std::string& role)
{
    std::optional<nlohmann::json> document;
    session.read(attachment, [&](const auto& store, const auto& txn, const auto&) {
        document = attachedDocument(store, txn, objectUuid, role);
    });
    return document;
}
}
