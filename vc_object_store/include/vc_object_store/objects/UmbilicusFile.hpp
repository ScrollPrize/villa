#pragma once
#include "vc_object_store/geometry/Geometry.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace vc::objects {
class ObjectStoreSession;
struct AttachmentIdentity;

// An umbilicus file (JSON: an array, or an object with "points" or
// "control_points", of [z, y, x] arrays or {z, y, x} objects, plus an optional
// "metadata" frame block; text/CSV: one "z, y, x" line per point) with the
// frame it declares. The optionals stay unset for unstamped files, which carry
// no statement about the grid their coordinates index.
struct UmbilicusFile {
    std::vector<geometry::Point> points;  // XYZ, in the file's order
    // metadata.voxelsize_um: voxel size, in µm, of the grid the coordinates
    // index. To express the umbilicus in a target grid, multiply the
    // coordinates by voxelSizeUm / targetVoxelSizeUm.
    std::optional<double> voxelSizeUm;
    // metadata.volume: volume store the umbilicus was annotated on. Provenance only.
    std::optional<std::string> volume;
    // metadata.volume_width / volume_height / volume_slices: the x, y and z
    // voxel counts of the grid the coordinates index, from which a rescale is
    // derived exactly where µm sizes are rounded.
    std::optional<int> volumeWidth, volumeHeight, volumeSlices;
    // One entry per metadata field that was present but rejected, so a typo is
    // distinguishable from a legacy file that never declared the field:
    // "<key>: <reason>, got <value-as-written>". Parsing still succeeds; callers
    // that decide a frame refuse a file with errors.
    std::vector<std::string> metadataErrors;
};
// Throws when the file cannot be read or holds no points.
UmbilicusFile readUmbilicusFile(const std::filesystem::path&);

// A volume grid: XYZ voxel counts and voxel size.
struct UmbilicusTargetGrid {
    std::array<double, 3> xyz{};
    std::optional<double> voxelSizeUm;
};
// How an umbilicus file binds to a volume of grid `target`, with `named` the
// grid of the volume its stamp names, when that is known: refused (with why),
// or the stored units per volume voxel the file states (`confirm` false), or
// a reading to confirm (inferred from the points, or nothing stated:
// storedUnitsPerVoxel unset).
struct UmbilicusBinding {
    std::optional<std::string> refusal;
    std::optional<double> storedUnitsPerVoxel;
    bool confirm = false;
    std::string frame;  // how the units were read, for provenance and prompts
};
UmbilicusBinding umbilicusBinding(const UmbilicusFile&, const UmbilicusTargetGrid& target,
                                  const std::optional<UmbilicusTargetGrid>& named);

// Imports `file`'s points as `volume`'s umbilicus, `storedUnitsPerVoxel` file
// units per volume voxel, named `name`, with `provenance` as meta.provenance;
// replaces the volume's umbilicus in one commit. Returns the new UUID.
std::string importVolumeUmbilicus(ObjectStoreSession&, const AttachmentIdentity&, const std::string& volume,
                                  const UmbilicusFile& file, double storedUnitsPerVoxel, const std::string& name,
                                  nlohmann::json provenance);
}
