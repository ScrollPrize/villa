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
    // The scroll's winding direction as the record states it
    // (meta.winding_handedness, +1 or -1); nullopt when it states none.
    std::optional<int> windingHandedness;
    // Who stated it (meta.winding_handedness_source): "catalog <sample>/<volume>"
    // for a direction taken from the open-data catalog, "manual" for one set
    // by hand; empty when the record does not say.
    std::string windingHandednessSource;
};
inline constexpr const char* kWindingHandednessMetaKey = "winding_handedness";
inline constexpr const char* kWindingHandednessSourceMetaKey = "winding_handedness_source";
// The winding direction an umbilicus record states (meta.winding_handedness),
// or nullopt. Throws std::invalid_argument for anything but +1 or -1: a
// mirrored map or solve must never come from a malformed value.
std::optional<int> umbilicusWindingHandedness(const ObjectRecord& record);
// Who stated it (meta.winding_handedness_source), empty when the record does not say.
std::string umbilicusWindingHandednessSource(const ObjectRecord& record);
// States the winding direction (+1 or -1) on an umbilicus record together with
// who states it: as operations of a checked request that expects the record at
// the version given, or in a write transaction. The store is where a
// direction looked up elsewhere (the open-data catalog) is kept, so it is not
// looked up again, and where one nobody else states is set by hand.
void setUmbilicusWindingHandednessRequest(TransactionRequest&, const ObjectRecord& umbilicus, int handedness,
                                          const std::string& source);
void setUmbilicusWindingHandedness(ObjectStore&, WriteTxn&, const ObjectRecord& umbilicus, int handedness,
                                   const std::string& source);
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
