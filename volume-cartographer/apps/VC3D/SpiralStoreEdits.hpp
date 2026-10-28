#pragma once

#include "SpiralPointCollectionEdit.hpp"

#include <vc_object_store/objects/AttachedDocument.hpp>
#include <vc_object_store/objects/ObjectStoreSession.hpp>

#include <nlohmann/json.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

class QuadSurface;

// Spiral drawings saved into the attached object store (stage 2's
// store-backed edits): a saved edit is a durable, checked revision of the
// store, synced by the session's hub worker; nothing here talks to a Spiral
// service. A fit sees the edits when its inputs are next updated.
namespace vc3d::spiral {

// The tag every brush patch carries, so a selection can name them.
inline constexpr const char* kBrushPatchTag = "spiral-brush";

// One brush patch: its geometry in the preview's frame, which is declared
// as `previewScale` stored units per voxel of the fit volume `volume`.
// `uuid` is the store object once saved; `remove` retires it.
struct BrushPatchSave {
    std::string uuid;  // empty: create a new surface
    std::string name;
    std::shared_ptr<QuadSurface> surface;
    bool remove = false;
};

// Save brush patches in one checked commit each: a new one becomes a surface
// object (tag kBrushPatchTag, with the surface's meta, which for a painted
// patch carries meta.source.spiral_patch_erode_cells 0), a replaced one gets a new
// geometry revision keeping its relationships as recorded (the next link
// refresh revisits them), a removed one is retired. Leases are held on a
// hosted replica. Returns each patch's UUID, in order; throws on the first
// failure (earlier patches stay saved).
std::vector<std::string> saveBrushPatches(vc::objects::ObjectStoreSession&, const vc::objects::AttachmentIdentity&,
                                          const std::string& volume, double previewScale,
                                          const std::vector<BrushPatchSave>& patches);

// A point-collection draft as the store keeps it, in reference-volume voxels:
// every point at its source position ("p" of its payload), kept points with
// their source ids (stable in the store), appended points with fresh ids
// after the highest id the source had, and each point's other payload fields
// (wind_a, creation_time) as its annotation. `source` is the object's current
// meta.source (empty for a new collection); winding_is_absolute is set from
// the draft's collection metadata and point_attributes dropped (annotations
// carry them).
vc::objects::AttachedCollection storedPointCollection(const EditablePclDraft& draft, const std::string& uuid,
                                                      const nlohmann::json& source);

// The store object a draft's source collection was read from
// (metadata.object_uuid of the fit's point-collection documents), if any.
std::optional<std::string> pointCollectionObject(const EditablePclDraft& draft);

// Save point-collection drafts and deletions in one checked commit (the
// store's writeVolumeCollections in the fit volume's voxels): replaced
// collections keep their identity and their points' ids, new ones are
// created, deleted ones retired. Returns the UUIDs written, in order.
std::vector<std::string> savePointCollections(vc::objects::ObjectStoreSession&, const vc::objects::AttachmentIdentity&,
                                              const std::string& volume,
                                              const std::vector<EditablePclDraft>& drafts);

} // namespace vc3d::spiral
