#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "vc/core/util/Umbilicus.hpp"

namespace vc::core::util {

    // The file's frame is read and judged by the store's umbilicus ingest
    // (vc_object_store/objects/UmbilicusFile.hpp); this adapts its verdict to
    // command-line tools that work from a target grid.
    // Outcome of loading an umbilicus file with a frame check. It mirrors
    // the resolver's Apply / Refuse / UseLegacy semantics, adapted to how
    // much the target grid may say about a declared frame. An authoritative
    // grid (the volume's own) may confirm or refute one; an inferred grid
    // (e.g. a command-line tool working from a surface bounding box) can do
    // neither for a grid-derived rescale — a surface patch's bounding box is
    // not its frame, and a partial surface can exactly mimic a downsampled
    // volume — so those warn and keep the legacy reading instead of
    // rescaling or refusing. Only an explicit coordinate conversion, a
    // stamped voxel size against the target's own, is applied under an
    // inferred grid. Under an authoritative grid a scale read off the
    // points is refused rather than applied: point inference only runs
    // when the stated frame could not be evaluated (a voxel-size-only
    // stamp with no target voxel size to compare it against), so rescaling
    // on it would treat an unchecked frame as checked.
    struct UmbilicusFrameLoad {
        // The loaded umbilicus. Empty only when the file was refused.
        std::optional<Umbilicus> umbilicus;
        // Why the file was refused. Empty when loading proceeded; callers
        // should print it and exit non-zero.
        std::string error;
        // A frame concern that did not stop loading. Empty when there is
        // none; callers should print it to stderr.
        std::string warning;
        // How the file's frame related to the working grid, for status
        // lines. Empty when the legacy reading was kept.
        std::string scaleDescription;
    };

    // How much the target grid passed to loadUmbilicusWithFrameCheck() may
    // say about a declared frame.
    enum class UmbilicusTargetGridAuthority {
        // The grid is inferred from data (e.g. surface bounding boxes): it
        // can neither confirm nor refute a grid-derived rescale. A surface
        // patch's bounding box is not its frame, and a partial surface can
        // exactly mimic a downsampled volume, so a declared frame implying
        // such a rescale warns and keeps the legacy reading. Only an
        // explicit coordinate conversion — a stamped voxel size against the
        // target's own voxel size — is applied.
        Inferred,
        // The grid is the authoritative volume grid: a declared frame that
        // does not fit it is refused outright, and so is one that cannot
        // be checked when the only scale comes from point inference — an
        // unchecked frame must not be rescaled as if it had been checked.
        Authoritative,
    };

    // Loads the umbilicus at `path` for a caller whose points live in the
    // frame of the working grid `targetGridXyz` (x, y, z voxel counts).
    // `volumeShape` ([z, y, x]) is the shape handed to the Umbilicus
    // constructor, exactly as Umbilicus::FromFile took it.
    //
    //   - malformed frame metadata   -> refused, with the errors listed;
    //   - stamped frame fits the grid -> points rescaled, scaleDescription
    //                                  set — but only under an Authoritative
    //                                  grid, or under an Inferred grid via an
    //                                  explicit stamped voxel-size
    //                                  conversion. A grid-derived rescale
    //                                  under an Inferred grid warns and keeps
    //                                  the legacy reading: surface bounds
    //                                  cannot confirm a rescale;
    //   - stamped frame does not fit  -> Inferred grids warn and keep the
    //                                  legacy reading (a mismatch proves
    //                                  nothing about the file); Authoritative
    //                                  grids refuse the file outright;
    //   - nothing stamped             -> legacy reading, as before.
    //
    // Throws when the file cannot be read, like FromFile.
    [[nodiscard]] UmbilicusFrameLoad loadUmbilicusWithFrameCheck(
        const std::filesystem::path& path,
        const std::array<double, 3>& targetGridXyz,
        const cv::Vec3i& volumeShape,
        UmbilicusTargetGridAuthority authority,
        std::optional<double> targetVoxelSizeUm = std::nullopt);

} // namespace vc::core::util
