#pragma once
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/StoreReader.hpp"

namespace vc::objects {
class ObjectStoreSession;
struct AttachmentIdentity;
// A volume has at most one umbilicus: the live umbilicus object whose
// coordinates map to it (bound to it, or through a registration).
struct VolumeUmbilicus {
    std::string uuid, name;
    ObjectVersion version; // the version read
    std::vector<geometry::Point> points; // in the volume's voxels
};
// The live umbilici that map to `volume` (one, unless the store is inconsistent).
std::vector<ObjectRecord> volumeUmbilici(const StoreReader&, const std::string& volume);
// The umbilicus of `volume`'s volume, its points in `volume`; nullopt when it
// has none. Throws when several map to it.
std::optional<VolumeUmbilicus> volumeUmbilicus(const StoreReader&, const VolumeVoxels& volume);
// The same, read from a session's attachment.
std::optional<VolumeUmbilicus> volumeUmbilicus(ObjectStoreSession&, const AttachmentIdentity&, const VolumeVoxels& volume);
// Makes `record` (an umbilicus whose meta.coordinates map to `volume`) with
// `geometry` the umbilicus of `volume`, retiring those it replaces. The
// caller's checked transaction pins volumeUmbilicusRevisions and expects
// record.uuid to be absent. Returns the UUID.
std::string setVolumeUmbilicus(ObjectStore&, WriteTxn&, const std::string& volume, ObjectRecord record,
                               const geometry::Umbilicus& geometry);
// The same as a checked request read through `reader`: expects the umbilici it
// retires at their current versions and record.uuid (required) absent. Guards
// the commit sequence before selecting, unless the request already has a guard;
// rebuild the request after a conflicting commit.
void setVolumeUmbilicusRequest(const StoreReader&, TransactionRequest&, const std::string& volume,
                               const ObjectRecord& record, const geometry::Umbilicus& geometry);
}
