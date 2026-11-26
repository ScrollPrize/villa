#include "vc_object_store/objects/VolumeUmbilicus.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/store/GeometryViews.hpp"
#include "internal/Database.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace vc::objects {
namespace {
ObjectSelection umbilici(const std::string& volume)
{
    ObjectSelection selection;
    selection.kind = "umbilicus";
    selection.volume = volume;
    return selection;
}

// An umbilicus record for `volume`: throws unless it maps to it.
void requireUmbilicusFor(const VolumeTransforms& transforms, const ObjectRecord& record, const std::string& volume)
{
    if (record.kind != "umbilicus") throw std::invalid_argument("setVolumeUmbilicus needs an umbilicus record");
    if (!transforms.toVolume(nlohmann::json::parse(record.meta), volume))
        throw std::invalid_argument("umbilicus " + record.name + " does not map to volume " + volume);
}
}

std::vector<ObjectRecord> volumeUmbilici(const StoreReader& reader, const std::string& volume)
{
    return reader.selectObjects(umbilici(volume));
}

std::optional<int> umbilicusWindingHandedness(const ObjectRecord& record)
{
    if (record.meta.empty()) return std::nullopt;
    const auto meta = nlohmann::json::parse(record.meta, nullptr, false);
    if (!meta.is_object()) return std::nullopt;
    const auto found = meta.find(kWindingHandednessMetaKey);
    if (found == meta.end() || found->is_null()) return std::nullopt;
    if (!found->is_number_integer())
        throw std::invalid_argument("umbilicus " + record.name + ": winding_handedness must be +1 or -1");
    const auto value = found->get<int>();
    if (value != 1 && value != -1)
        throw std::invalid_argument("umbilicus " + record.name + ": winding_handedness must be +1 or -1");
    return value;
}

std::string umbilicusWindingHandednessSource(const ObjectRecord& record)
{
    if (record.meta.empty()) return {};
    const auto meta = nlohmann::json::parse(record.meta, nullptr, false);
    if (!meta.is_object()) return {};
    const auto found = meta.find(kWindingHandednessSourceMetaKey);
    return found != meta.end() && found->is_string() ? found->get<std::string>() : std::string{};
}

namespace {
void requireWindingStatement(const ObjectRecord& umbilicus, int handedness, const std::string& source)
{
    if (umbilicus.kind != "umbilicus") throw std::invalid_argument("a winding direction is stated on an umbilicus record");
    if (handedness != 1 && handedness != -1) throw std::invalid_argument("winding_handedness must be +1 or -1");
    if (source.empty()) throw std::invalid_argument("a winding direction needs who states it");
}
}

void setUmbilicusWindingHandednessRequest(TransactionRequest& request, const ObjectRecord& umbilicus, int handedness,
                                          const std::string& source)
{
    requireWindingStatement(umbilicus, handedness, source);
    request.expectedObjects.insert_or_assign(umbilicus.uuid, umbilicus.reference());
    request.operations.push_back(SetMetaOp{umbilicus.uuid, kWindingHandednessMetaKey, nlohmann::json(handedness).dump()});
    request.operations.push_back(SetMetaOp{umbilicus.uuid, kWindingHandednessSourceMetaKey, nlohmann::json(source).dump()});
}

void setUmbilicusWindingHandedness(ObjectStore& store, WriteTxn& txn, const ObjectRecord& umbilicus, int handedness,
                                   const std::string& source)
{
    requireWindingStatement(umbilicus, handedness, source);
    store.setMeta(txn, umbilicus.uuid, kWindingHandednessMetaKey, nlohmann::json(handedness).dump());
    store.setMeta(txn, umbilicus.uuid, kWindingHandednessSourceMetaKey, nlohmann::json(source).dump());
}

std::optional<VolumeUmbilicus> volumeUmbilicus(const StoreReader& reader, const VolumeVoxels& volume)
{
    const auto records = volumeUmbilici(reader, volume.volume);
    if (records.empty()) return std::nullopt;
    if (records.size() > 1) {
        std::string names;
        for (const auto& record : records) names += " " + (record.name.empty() ? record.uuid : record.name);
        throw std::runtime_error("several umbilici map to volume " + volume.volume + ":" + names);
    }
    const auto& record = records.front();
    const auto head = reader.selectedObject(record.uuid, {}, {}, record.reference());
    if (!head || !head->geometryRevision) throw std::runtime_error("umbilicus " + record.name + " has no geometry");
    // The geometry of the version read, so points and version agree.
    const auto canonical = GeometrySnapshot(reader).umbilicus(*head);
    VolumeUmbilicus result{record.uuid, record.name, head->reference(),
                           canonical.viewTransformedAt(volume.volume, volume.scale).points("points", 0, canonical.size()),
                           umbilicusWindingHandedness(record), umbilicusWindingHandednessSource(record)};
    return result;
}

std::optional<VolumeUmbilicus> volumeUmbilicus(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                                               const VolumeVoxels& volume)
{
    std::optional<VolumeUmbilicus> umbilicus;
    session.readStore(attachment, [&](const StoreReader& reader, const auto&) { umbilicus = volumeUmbilicus(reader, volume); });
    return umbilicus;
}

std::string setVolumeUmbilicus(ObjectStore& store, WriteTxn& txn, const std::string& volume, ObjectRecord record,
                               const geometry::Umbilicus& geometry)
{
    detail::Savepoint save(ObjectStoreImplAccess::conn(txn));
    requireUmbilicusFor(store.volumeTransforms(txn), record, volume);
    for (const auto& existing : store.selectObjects(txn, umbilici(volume))) store.retireObject(txn, existing.uuid);
    const auto uuid = store.upsertObject(txn, record);
    store.putGeometry(txn, uuid, geometry::encode(geometry));
    save.release();
    return uuid;
}

void setVolumeUmbilicusRequest(const StoreReader& reader, TransactionRequest& request, const std::string& volume,
                               const ObjectRecord& record, const geometry::Umbilicus& geometry)
{
    if (record.uuid.empty()) throw std::invalid_argument("setVolumeUmbilicusRequest needs the record's UUID");
    // Guard before reading: new umbilici or registrations can change the
    // selection without changing any of the object versions it returns.
    // Preserve an earlier guard if this request already depends on a snapshot.
    if (!request.expectedCommitSequence) request.expectedCommitSequence = reader.status().commitSequence;
    requireUmbilicusFor(reader.volumeTransforms(), record, volume);
    for (const auto& existing : volumeUmbilici(reader, volume)) {
        expectObject(request, existing.reference());
        request.operations.push_back(RetireObjectOp{existing.uuid});
    }
    request.expectedObjects[record.uuid] = std::nullopt;
    request.operations.push_back(CreateObjectOp{objectRecordJson(record)});
    request.operations.push_back(PutGeometryOp{record.uuid, geometry::encode(geometry), {}});
}
}
