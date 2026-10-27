#include "vc_object_store/objects/ObjectCopy.hpp"
#include "vc_object_store/Uuid.hpp"
#include <nlohmann/json.hpp>
#include <limits>

namespace vc::objects {
ObjectCopy readObjectCopy(const ObjectStore& store, const Txn& txn, const std::string& objectUuid,
                          const std::optional<std::string>& copyUuid)
{
    const auto object = store.objectByUuid(txn, objectUuid);
    if (!object) throw std::invalid_argument("object to copy not found: " + objectUuid);
    const auto head = store.currentObjectRevision(txn, object->uuid);
    const auto geometry = store.currentGeometry(txn, object->uuid);
    ObjectCopy copy;
    auto& record = copy.record;
    record.uuid = copyUuid ? *copyUuid : util::newUuid();
    record.kind = object->kind;
    record.name = object->name;
    record.tags = object->tags;
    record.formatVersion = object->formatVersion;
    record.fiber = object->fiber;
    record.surface = object->surface;
    record.track = object->track;
    auto meta = nlohmann::json::parse(object->meta);
    meta["copied_from"] = {{"store_uuid", store.uuid(txn)}, {"object_uuid", object->uuid},
        {"object_revision", head ? nlohmann::json(head->uuid) : nlohmann::json()},
        {"geometry_revision", geometry ? nlohmann::json(geometry->uuid) : nlohmann::json()}};
    if (geometry) {
        copy.geometry = store.readGeometryData(txn, geometry->uuid, std::numeric_limits<uint64_t>::max());
        // Per-point annotations are consumed from meta.source when the geometry is put.
        auto source = meta.value("source", nlohmann::json::object());
        for (const auto& array : copy.geometry->arrays) {
            if (copy.geometry->kind == "surface" ||
                (array.name != "points" && array.name != "control_points" && array.name != "line_points")) continue;
            const auto key = copy.geometry->kind == "point_collection" ? std::string("point_attributes") : array.name + "_attributes";
            source[key] = nlohmann::json::parse(store.geometryPointAnnotations(txn, geometry->uuid, array.name));
        }
        if (!source.empty()) meta["source"] = std::move(source);
    }
    record.meta = meta.dump();
    return copy;
}

std::string writeObjectCopy(ObjectStore& store, WriteTxn& txn, const ObjectCopy& copy)
{
    const auto uuid = store.upsertObject(txn, copy.record);
    if (copy.geometry) store.putGeometry(txn, uuid, *copy.geometry);
    return uuid;
}
}
