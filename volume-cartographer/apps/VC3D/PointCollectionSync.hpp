#pragma once

#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/objects/ObjectStoreSession.hpp>

#include <QObject>
#include <QTimer>

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

class CState;

namespace vc3d {

// Keeps CState's point collections without a surface anchor (a volume's own;
// anchored ones are a surface's corrections) in the object store: they are
// loaded from the current volume's point_collection objects when the store
// or volume changes, or when someone else commits to them, and edits are
// written back in debounced transactions (one per burst).
class PointCollectionSync : public QObject {
    Q_OBJECT
public:
    explicit PointCollectionSync(CState* state);

    // Write pending edits now; retain them and return false on failure.
    bool flush();
    // Flush, then drop the volume's collections from memory, leaving the store
    // as it is (closing the project).
    void unload();

private:
    void reload();
    void markChanged(uint64_t collectionId);
    void markRemoved(uint64_t collectionId);
    bool anchored(uint64_t collectionId) const;
    void onStoreEvent(const vc::objects::StoreSessionEvent& event);
    void noteRevisions(const std::vector<std::string>& uuids);

    CState* state_;
    QTimer debounce_;
    // What the loaded collections belong to, and the voxels they are in.
    std::optional<vc::objects::AttachmentIdentity> attachment_;
    vc::objects::VolumeVoxels volume_;
    // Runtime id to UUID of the volume's collections in memory.
    std::map<uint64_t, std::string> uuids_;
    std::set<uint64_t> changed_;
    std::set<std::string> removed_;
    // The object revision of each collection as this sync last wrote or read
    // it: a commit leaving another is someone else's.
    std::map<std::string, std::string> revisions_;
    // Installing from the store: the collection's change hooks are not edits.
    bool installing_ = false;
};

} // namespace vc3d
