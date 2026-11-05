#include "vc_object_store/objects/VolumeUmbilicus.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/store/GeometryViews.hpp"
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

RevisionSelection volumeUmbilicusRevisions(const ObjectStore& store, const Txn& txn, const std::string& volume)
{
    return store.currentRevisions(txn, umbilici(volume));
}

std::vector<ObjectRecord> volumeUmbilici(const StoreReader& reader, const std::string& volume)
{
    return reader.selectObjects(umbilici(volume));
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
    const auto head = reader.currentObject(record.uuid);
    if (!head || !head->geometryRevision) throw std::runtime_error("umbilicus " + record.name + " has no geometry");
    // The geometry of the version read, so points and version agree.
    const auto canonical = GeometrySnapshot(reader).umbilicus(record.uuid, *head->geometryRevision);
    VolumeUmbilicus result{record.uuid, record.name, {reader.status().storeUuid, head->generation, record.uuid, head->version},
                           canonical.viewTransformedAt(volume.volume, volume.scale).points("points", 0, canonical.size())};
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
    requireUmbilicusFor(store.volumeTransforms(txn), record, volume);
    for (const auto& [uuid, revision] : volumeUmbilicusRevisions(store, txn, volume)) store.retireObject(txn, uuid);
    const auto uuid = store.upsertObject(txn, record);
    store.putGeometry(txn, uuid, geometry::encode(geometry));
    return uuid;
}

void setVolumeUmbilicusRequest(const StoreReader& reader, TransactionRequest& request, const std::string& volume,
                               const ObjectRecord& record, const geometry::Umbilicus& geometry)
{
    if (record.uuid.empty()) throw std::invalid_argument("setVolumeUmbilicusRequest needs the record's UUID");
    requireUmbilicusFor(reader.volumeTransforms(), record, volume);
    for (const auto& existing : volumeUmbilici(reader, volume)) {
        request.expectedObjects[existing.uuid] = existing.version;
        request.operations.push_back(RetireObjectOp{existing.uuid});
    }
    request.expectedObjects[record.uuid] = std::nullopt;
    request.operations.push_back(CreateObjectOp{objectRecordJson(record)});
    request.operations.push_back(PutGeometryOp{record.uuid, geometry::encode(geometry), {}});
}
}
