#pragma once

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <vc_object_store/objects/ObjectStoreSession.hpp>
#include <vc_object_store/objects/UmbilicusFile.hpp>

#include "vc/core/types/VolumePkg.hpp"

class CState;

namespace vc3d {

// An umbilicus file read and fitted to the current volume, ready to import into
// the store umbilici are routed to, as the umbilicus of the annotation volume.
struct UmbilicusImportPlan {
    std::filesystem::path file;
    vc::objects::UmbilicusFile umbilicus;
    // binding.refusal says why the file does not fit; binding.confirm asks for
    // the units per voxel to be confirmed (binding.storedUnitsPerVoxel is the inference).
    vc::objects::UmbilicusBinding binding;
    vc::objects::AttachmentIdentity store;
    vc::project::AnnotationVolume target;
    // The volume already has an umbilicus, which the import replaces.
    bool replaces{false};
};

// Thrown by planUmbilicusImport when `file` is not a readable umbilicus file.
struct UmbilicusFileUnreadable : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Throws UmbilicusFileUnreadable when `file` is not an umbilicus file, and
// another exception when no project, volume or umbilicus store is selected or
// the store cannot be read.
UmbilicusImportPlan planUmbilicusImport(const CState&, const std::filesystem::path& file);
// Imports at `storedUnitsPerVoxel` file units per current-volume voxel,
// replacing the volume's umbilicus in one commit. Returns the new UUID.
std::string importUmbilicus(const CState&, const UmbilicusImportPlan&, double storedUnitsPerVoxel);

// The live umbilicus objects of the annotation volume in the routed umbilicus store.
struct AnnotationUmbilici {
    vc::objects::AttachmentIdentity store;
    std::string volume;
    std::vector<std::string> uuids;
};
// nullopt without a project, volume or umbilicus store.
std::optional<AnnotationUmbilici> annotationUmbilici(const CState&);
// Retires them in one transaction, for everyone sharing the store; returns their UUIDs.
std::vector<std::string> retireUmbilici(const CState&, const AnnotationUmbilici&);

// The scroll's winding direction in the annotation volume, as its umbilicus in
// the routed umbilicus store states it (+1: the winding grows with the angle
// about the umbilicus, -1: against it), and who stated it: "catalog
// <sample>/<volume>", "manual", or empty. `problem` says why there is none to
// read (no umbilicus, several).
struct WindingDirection {
    std::string volume;
    std::optional<int> handedness;
    std::string source, problem;
};
// nullopt without a project, volume or umbilicus store.
std::optional<WindingDirection> windingDirection(const CState&);
// States it on that umbilicus. Throws when there is not exactly one, or the
// store cannot be written.
void stateWindingDirection(const CState&, int handedness, const std::string& source);
// The direction the open-data catalog gives the current volume, with the
// catalog's name of it ("<sample>/<volume>"); nullopt where the catalog does
// not orient the volume.
std::optional<std::pair<int, std::string>> catalogWindingDirection(const CState&);
// Where the store states none and the catalog gives one: stored on the
// umbilicus, so it is looked up once. Returns what was stored, nullopt when
// nothing was (already stated, no catalog direction, no umbilicus). Throws
// when the store cannot be written.
std::optional<WindingDirection> keepCatalogWindingDirection(const CState&);

}  // namespace vc3d
