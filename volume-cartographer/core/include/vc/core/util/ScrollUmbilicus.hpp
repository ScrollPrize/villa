#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "vc/core/util/Umbilicus.hpp"

class VolumePkg;

namespace vc::core::util {

    struct ScrollUmbilicusResolution {
        // The chosen file; empty when nothing could be resolved.
        std::filesystem::path path;
        // Contents of `path`, when it is non-empty.
        UmbilicusFileInfo info;
        // The distinct candidates found when a directory search turned up more
        // than one, in which case `path` is left empty.
        std::vector<std::filesystem::path> ambiguous;
        // Human-readable reason why `path` is empty.
        std::string error;
    };

    // Resolves the umbilicus a package's consumers should use.
    //
    // The project's "umbilicus" field is authoritative: when it is set at all,
    // that location is the answer and no directory search ever happens — not
    // even when the location cannot be used. A configured location that
    // umbilicusPath() cannot turn into a local file (s3://, http://, ...) is
    // reported as an unsupported-location error, distinct from the
    // does-not-exist error a missing local file gets.
    //
    // Only an empty field proceeds to discovery: the package root, plus both
    // the parent and the grandparent of every attached segments entry, are
    // searched for umbilicus.json, then estimated_umbilicus.json. Searching
    // grandparents covers the <volpkg>/paths/<segment> layout, whose
    // <volpkg>/umbilicus.json a parent-only search misses. The result is only
    // accepted when all roots agree on a single file; one file reachable
    // through several roots is deduplicated and stays unambiguous.
    //
    // Frame metadata is reported as the file declares it, and unstamped files
    // are returned as such for callers to decide about — but a file whose
    // metadata is present and malformed (UmbilicusFileInfo::metadataErrors) is
    // refused outright, with the errors listed in `error`, so that a typo can
    // never be mistaken downstream for a legacy unstamped file.
    [[nodiscard]] ScrollUmbilicusResolution resolveScrollUmbilicus(
        const VolumePkg& pkg);

    // How a scale was arrived at, because callers trust the three differently:
    // the stamped ones are stated by the file, the inferred one is this code's
    // best reading of it.
    enum class UmbilicusScaleSource {
        StampedDimensions,
        StampedVoxelSize,
        InferredFromGrid,
    };

    struct UmbilicusScale {
        double factor = 1.0;
        UmbilicusScaleSource source = UmbilicusScaleSource::StampedDimensions;
        // Plain-text account of the derivation, for logs and status lines.
        std::string description;
    };

    // The factor that carries an umbilicus file's coordinates into the frame
    // whose extent is targetGridXyz (x, y, z voxel counts) — the frame the
    // caller's own coordinates live in, which is not necessarily any single
    // volume's own grid.
    //
    // Tried in descending order of trust: the stamped grid dimensions against
    // the target grid (exact integers, no micrometres involved, and rejected
    // outright if the three axis ratios disagree); the stamped voxel size
    // against targetVoxelSizeUm; and finally which power-of-two downsample of
    // the target grid the points fit inside and nearly fill, which is a reading
    // of the file rather than a statement by it. Empty when none of the three
    // applies, which callers must treat as "frame unknown" rather than 1.0.
    //
    // Deliberately says nothing about registration: this answers which grid the
    // numbers index, not which pose that grid was in. A file needing both is
    // outside what the format can currently express.
    [[nodiscard]] std::optional<UmbilicusScale> deriveUmbilicusScale(
        const UmbilicusFileInfo& info,
        const std::array<double, 3>& targetGridXyz,
        std::optional<double> targetVoxelSizeUm);

} // namespace vc::core::util
