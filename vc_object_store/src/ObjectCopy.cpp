#include "vc_object_store/objects/ObjectCopy.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/Uuid.hpp"
#include "internal/Database.hpp"
#include <nlohmann/json.hpp>
#include <limits>

namespace vc::objects {
ObjectCopy readObjectCopy(const StoreReader& reader, const std::string& objectUuid,
                          const std::optional<std::string>& copyUuid)
{
    if (reader.status().extensionsPresent)
        throw std::runtime_error("object copy: registered extensions require a database-preserving copy; refusing silent data loss");
    const auto head = reader.currentObject(util::canonicalUuid(objectUuid));
    if (!head) throw std::invalid_argument("object to copy not found: " + objectUuid);
    const auto& object = head->record;
    const auto geometry = head->geometryRevision;
    ObjectCopy copy;
    auto& record = copy.record;
    record.uuid = copyUuid ? util::canonicalUuid(*copyUuid) : util::newUuid();
    record.kind = object.kind;
    record.name = object.name;
    record.tags = object.tags;
    record.formatVersion = object.formatVersion;
    record.fiber = object.fiber;
    record.surface = object.surface;
    record.track = object.track;
    auto meta = nlohmann::json::parse(object.meta);
    if (meta.contains("vcobj_extensions"))
        throw std::runtime_error("object copy: manual extension fields require an extension-aware copy");
    meta["copied_from"] = {{"store_uuid", head->reference().storeUuid}, {"object_uuid", object.uuid},
        {"generation", head->generation}, {"version", head->version},
        {"geometry_revision", geometry ? nlohmann::json(*geometry) : nlohmann::json()}};
    if (geometry) {
        copy.geometry = reader.readGeometryData(*geometry, std::numeric_limits<uint64_t>::max());
        // Per-point annotations are consumed from meta.source when the geometry is put.
        auto source = meta.value("source", nlohmann::json::object());
        for (const auto& array : copy.geometry->arrays) {
            if (copy.geometry->kind == "surface" ||
                (array.name != "points" && array.name != "control_points" && array.name != "line_points")) continue;
            const auto key = copy.geometry->kind == "point_collection" ? std::string("point_attributes") : array.name + "_attributes";
            source[key] = reader.geometryPointAnnotations(*geometry, array.name);
        }
        if (!source.empty()) meta["source"] = std::move(source);
    }
    record.meta = meta.dump();
    return copy;
}

std::vector<TransactionOperation> objectCopyOperations(const ObjectCopy& copy)
{
    std::vector<TransactionOperation> operations{CreateObjectOp{objectRecordJson(copy.record)}};
    if (copy.geometry) operations.push_back(PutGeometryOp{copy.record.uuid, *copy.geometry, {}, false});
    return operations;
}

std::string writeObjectCopy(ObjectStore& store, WriteTxn& txn, const ObjectCopy& copy)
{
    detail::Savepoint operation(ObjectStoreImplAccess::conn(txn));
    const auto uuid = store.upsertObject(txn, copy.record);
    if (copy.geometry) store.putGeometry(txn, uuid, *copy.geometry);
    operation.release();
    return uuid;
}
}
