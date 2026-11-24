#pragma once

#include "SpiralPointCollectionEdit.hpp"

#include <vc_object_store/objects/AttachedDocument.hpp>
#include <vc_object_store/objects/ObjectStoreSession.hpp>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

class QuadSurface;

// Spiral drawings saved into the attached object store (stage 2's
// store-backed edits): a saved edit is a checked commit of the store (a local
// file or a served store); nothing here talks to a Spiral service. A fit sees
// the edits when its inputs are next updated.
namespace vc3d::spiral {

// The tag every brush patch carries, so a selection can name them.
inline constexpr const char* kBrushPatchTag = "spiral-brush";

// One brush patch: its geometry in the preview's frame, which is declared
// as `previewScale` stored units per voxel of the fit volume `volume`.
// `uuid` is the store object once saved, and `reference` its complete selected
// identity (owner, lineage, UUID and version). `remove`
// retires it.
struct BrushPatchSave {
    std::string uuid;  // empty: create a new surface
    vc::objects::ObjectVersion reference;  // required with `uuid`
    std::string name;
    std::shared_ptr<QuadSurface> surface;
    bool remove = false;
};

// A saved patch: its object and the reference the save wrote (empty when it was
// retired), which the next edit of it starts from, and the request UUID of
// the commit that wrote it (its receipt).
struct SavedBrushPatch {
    std::string uuid;
    vc::objects::ObjectVersion reference;
    std::string requestUuid;
};

// A save refused because another editor saved the object after the version
// the edit started from; `current` is their complete reference (empty when gone).
// Nothing was written: saving again from `current` replaces theirs.
struct StoreObjectChanged : std::runtime_error {
    StoreObjectChanged(const std::string& message, std::string uuid, vc::objects::ObjectVersion current)
        : std::runtime_error(message), uuid(std::move(uuid)), current(current) {}
    std::string uuid;
    vc::objects::ObjectVersion current;
};

// Save brush patches in one checked commit each: a new one becomes a surface
// object (tag kBrushPatchTag, with the surface's meta, which for a painted
// patch carries meta.source.spiral_patch_erode_cells 0), a replaced one gets a new
// geometry revision keeping its relationships as recorded (the next link
// refresh revisits them), a removed one is retired. A replaced or removed
// patch is expected at the version its edit started from: when another
// editor saved it since, the save is refused (StoreObjectChanged) instead of
// overwriting theirs. Returns each patch's saved state, its version from the
// commit's receipt; throws on the first failure (earlier patches stay saved).
std::vector<SavedBrushPatch> saveBrushPatches(vc::objects::ObjectStoreSession&, const vc::objects::AttachmentIdentity&,
                                              const std::string& volume, double previewScale,
                                              const std::vector<BrushPatchSave>& patches);

// A point-collection draft as the store keeps it, in reference-volume voxels:
// every point at its source position ("p" of its payload), kept points with
// their source ids (stable in the store), appended points marked for the shared
// retained-ID allocator at save time, and each point's other payload fields
// (wind_a, creation_time) as its annotation. `source` is the object's current
// meta.source (empty for a new collection); winding_is_absolute is set from
// the draft's collection metadata and point_attributes dropped (annotations
// carry them).
vc::objects::AttachedCollection storedPointCollection(const EditablePclDraft& draft, const std::string& uuid,
                                                      const nlohmann::json& source);

// The store object a draft's source collection was read from
// (the reference retained when its document was loaded), if any.
std::optional<std::string> pointCollectionObject(const EditablePclDraft& draft);

// Name the store object (and the version of it) a draft edits: a new
// collection saved once is replaced, not created again, by its next save.
void setPointCollectionObject(EditablePclDraft& draft, const vc::objects::ObjectVersion& reference);

// A saved draft: its object and the reference the save wrote (empty when it was
// retired, or when a deleted draft had no object), and the request UUID of
// the commit that wrote it (its receipt; every draft of one save shares it).
using SavedPointCollection = SavedBrushPatch;

// Save point-collection drafts and deletions in one checked commit (the
// store's writeVolumeCollections in the fit volume's voxels): replaced
// collections keep their identity and their points' ids, new ones are
// created, deleted ones retired. A replaced or deleted collection is expected
// at the full typed identity its draft was read at: when
// another editor changed or retired it since, the commit is refused
// (StoreObjectChanged) instead of overwriting their edit. Returns one result
// per draft, in order.
std::vector<SavedPointCollection> savePointCollections(vc::objects::ObjectStoreSession&,
                                                       const vc::objects::AttachmentIdentity&,
                                                       const std::string& volume,
                                                       const std::vector<EditablePclDraft>& drafts);

} // namespace vc3d::spiral
