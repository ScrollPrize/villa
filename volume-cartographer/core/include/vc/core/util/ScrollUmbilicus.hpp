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

    // The single factor that carries `stampedXyz` voxel counts onto `targetXyz`
    // voxel counts, or nothing when no one factor explains all three axes.
    //
    // Downsampling by an integer factor n maps a count c to floor(c/n) or
    // ceil(c/n) and to nothing else, so a candidate factor is tested exactly:
    // every axis must land on one of those two values. There is no tolerance to
    // tune and no averaging. An earlier version accepted any spread of axis
    // ratios under 2% -- 654 voxels on a 32,696-voxel axis -- and returned their
    // mean, a factor matching none of them.
    //
    // Both directions are tried, so the stamped grid may be the coarser or the
    // finer of the two. What this cannot do is pin a *fractional* rescale: voxel
    // counts alone are consistent with a range of non-integer factors, so a file
    // rescaled by, say, 1.5 is not identified here and must state its voxel size
    // instead. Non-power-of-two integer factors are fine.
    //
    // Several candidates can only survive on grids of a handful of voxels, where
    // the rounding windows overlap; that yields nothing rather than a guess.
    [[nodiscard]] std::optional<double> uniformRescaleFactor(
        const std::array<double, 3>& stampedXyz,
        const std::array<double, 3>& targetXyz);

    // The factor that carries an umbilicus file's coordinates into the frame
    // whose extent is targetGridXyz (x, y, z voxel counts) — the frame the
    // caller's own coordinates live in, which is not necessarily any single
    // volume's own grid.
    //
    // Tried in descending order of trust: the stamped grid dimensions against
    // the target grid through uniformRescaleFactor() (integer counts, no
    // micrometres involved, and rejected outright when no one factor explains
    // all three axes); the stamped voxel size
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

    // Whether the file itself declares the grid its coordinates index, and by
    // which key. Dimensions outrank voxel size: exact integer counts, no
    // rounding, and they express rescales no µm figure can.
    //
    // This is deliberately separate from whether a scale could be *derived*.
    // deriveUmbilicusScale() returns nothing both for a file that says nothing
    // and for a file whose statement does not fit the target, and those two
    // deserve opposite treatment — hence a predicate that answers only the
    // first half. Malformed values never reach here: they populate
    // UmbilicusFileInfo::metadataErrors and the resolver refuses the file.
    struct UmbilicusFrameClaim {
        bool dimensions = false;   // complete, positive triplet
        bool voxelSize = false;
        [[nodiscard]] bool any() const { return dimensions || voxelSize; }
    };

    [[nodiscard]] UmbilicusFrameClaim umbilicusFrameClaim(
        const UmbilicusFileInfo& info);

    // What a consumer should do with a resolved umbilicus file.
    enum class UmbilicusLoadAction {
        // A scale was derived; carry the points into the target frame.
        Apply,
        // The file states a frame that does not fit the target, or states one that
        // could not be checked because there is no target grid. Refusing is the
        // point: a frame applied confidently but wrongly is worse than none, and
        // silently reinterpreting it is what this replaced. "Cannot evaluate" and
        // "does not match" differ in their diagnostics, not in whether the file is
        // used.
        Refuse,
        // The file states nothing usable, so a consumer may fall back to
        // whatever reading it used before frames were declarable.
        UseLegacy,
    };

    // Extracted from the branch it drives so all three arms are testable
    // without a volume: the consumer that needs it lives behind a package, a
    // loaded volume and a registration transform.
    //
    // haveTargetGrid must be false when the caller cannot say what frame it
    // wants. Refusing then would blame the file for the caller's own missing
    // information — and a stated frame that could not even be compared is not
    // a conflict.
    [[nodiscard]] UmbilicusLoadAction decideUmbilicusLoadAction(
        const std::optional<UmbilicusScale>& scale,
        const UmbilicusFrameClaim& claim,
        bool haveTargetGrid);

} // namespace vc::core::util
