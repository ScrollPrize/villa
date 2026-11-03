#include "vc_object_store/objects/VolumeUmbilicus.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
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
}

RevisionSelection volumeUmbilicusRevisions(const ObjectStore& store, const Txn& txn, const std::string& volume)
{
    return store.currentRevisions(txn, umbilici(volume));
}

std::optional<VolumeUmbilicus> volumeUmbilicus(const ObjectStore& store, const Txn& txn, const VolumeVoxels& volume)
{
    const auto records = store.selectObjects(txn, umbilici(volume.volume));
    if (records.empty()) return std::nullopt;
    if (records.size() > 1) {
        std::string names;
        for (const auto& record : records) names += " " + (record.name.empty() ? record.uuid : record.name);
        throw std::runtime_error("several umbilici map to volume " + volume.volume + ":" + names);
    }
    const auto& record = records.front();
    if (!store.currentGeometry(txn, record.uuid))
        throw std::runtime_error("umbilicus " + record.name + " has no geometry");
    const auto canonical = GeometrySnapshot(store, txn).umbilicus(record.uuid);
    VolumeUmbilicus result{record.uuid, record.name, store.currentObjectRevision(txn, record.uuid)->uuid,
                           canonical.viewTransformedAt(volume.volume, volume.scale).points("points", 0, canonical.size())};
    return result;
}

std::optional<VolumeUmbilicus> volumeUmbilicus(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                                               const VolumeVoxels& volume)
{
    std::optional<VolumeUmbilicus> umbilicus;
    session.read(attachment, [&](const auto& store, const auto& txn, const auto&) {
        umbilicus = volumeUmbilicus(store, txn, volume);
    });
    return umbilicus;
}

std::string setVolumeUmbilicus(ObjectStore& store, WriteTxn& txn, const std::string& volume, ObjectRecord record,
                               const geometry::Umbilicus& geometry)
{
    if (record.kind != "umbilicus") throw std::invalid_argument("setVolumeUmbilicus needs an umbilicus record");
    if (!store.volumeTransforms(txn).toVolume(nlohmann::json::parse(record.meta), volume))
        throw std::invalid_argument("umbilicus " + record.name + " does not map to volume " + volume);
    for (const auto& [uuid, revision] : volumeUmbilicusRevisions(store, txn, volume)) store.retireObject(txn, uuid);
    const auto uuid = store.upsertObject(txn, record);
    store.putGeometry(txn, uuid, geometry::encode(geometry));
    return uuid;
}
}
