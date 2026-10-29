#pragma once
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"

namespace vc::objects {
class ObjectStoreSession;
struct AttachmentIdentity;
// A volume has at most one umbilicus: the live umbilicus object whose
// coordinates map to it (bound to it, or through a registration).
struct VolumeUmbilicus {
    std::string uuid, name, objectRevision;
    std::vector<geometry::Point> points; // in the volume's voxels
};
// The current revisions of the umbilici that map to `volume`: the pins of an
// edit that replaces or removes them.
RevisionSelection volumeUmbilicusRevisions(const ObjectStore&, const Txn&, const std::string& volume);
// The umbilicus of `volume`'s volume, its points in `volume`; nullopt when it
// has none. Throws when several map to it.
std::optional<VolumeUmbilicus> volumeUmbilicus(const ObjectStore&, const Txn&, const VolumeVoxels& volume);
// The same, read from a session's attachment.
std::optional<VolumeUmbilicus> volumeUmbilicus(ObjectStoreSession&, const AttachmentIdentity&, const VolumeVoxels& volume);
// Makes `record` (an umbilicus whose meta.coordinates map to `volume`) with
// `geometry` the umbilicus of `volume`, retiring those it replaces. The
// caller's checked transaction pins volumeUmbilicusRevisions and expects
// record.uuid to be absent. Returns the UUID.
std::string setVolumeUmbilicus(ObjectStore&, WriteTxn&, const std::string& volume, ObjectRecord record,
                               const geometry::Umbilicus& geometry);
}
