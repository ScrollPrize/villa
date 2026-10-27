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

// The single integer factor n (or 1/m), up to 64, that carries the stamped
// voxel counts onto the target's (each axis floor or ceil of the other divided
// by it), else nothing: the counts do not identify one.
std::optional<double> uniformRescaleFactor(const std::array<double, 3>& stampedXyz,
                                           const std::array<double, 3>& targetXyz);

enum class UmbilicusScaleSource {
    StampedDimensions,
    StampedVoxelSize,
    InferredFromGrid,
};
// The factor that carries the file's coordinates into a target grid's voxels.
struct UmbilicusScale {
    double factor = 1.0;
    UmbilicusScaleSource source = UmbilicusScaleSource::StampedDimensions;
    std::string description;
};
// From the stamped dimensions against the target grid (XYZ counts), else the
// stamped voxel size against the target's, else (nothing stamped) the one
// power-of-two grid the points nearly fill. Nothing when no reading applies.
std::optional<UmbilicusScale> deriveUmbilicusScale(const UmbilicusFile&, const std::array<double, 3>& targetGridXyz,
                                                   std::optional<double> targetVoxelSizeUm);

// What frame the file states: a complete positive dimension triplet, a voxel size.
struct UmbilicusFrameClaim {
    bool dimensions = false;
    bool voxelSize = false;
    [[nodiscard]] bool any() const { return dimensions || voxelSize; }
};
UmbilicusFrameClaim umbilicusFrameClaim(const UmbilicusFile&);

// Why the file's stamp is wrong about the volume it names (that volume's grid
// and voxel size), or nothing when it is consistent with it.
std::optional<std::string> umbilicusStampContradiction(const UmbilicusFile&,
                                                       const std::array<double, 3>& namedVolumeGridXyz,
                                                       std::optional<double> namedVolumeVoxelSizeUm);

enum class UmbilicusLoadAction {
    Apply,
    Refuse,
    UseLegacy,
};
// A contradicted stamp refuses; without a target grid a stated frame refuses
// and an unstated one keeps the legacy reading; a derived scale applies; a
// stated frame that derived none refuses.
UmbilicusLoadAction decideUmbilicusLoadAction(const std::optional<UmbilicusScale>&, const UmbilicusFrameClaim&,
                                              bool haveTargetGrid, bool stampContradicted = false);

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
