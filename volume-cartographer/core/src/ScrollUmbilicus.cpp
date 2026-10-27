#include "vc/core/util/ScrollUmbilicus.hpp"

#include <vc_object_store/objects/UmbilicusFile.hpp>

#include <algorithm>
#include <optional>
#include <limits>
#include <cmath>
#include <array>
#include <vector>
#include <exception>
#include <system_error>

namespace vc::core::util {

using namespace vc::objects;

UmbilicusFrameLoad loadUmbilicusWithFrameCheck(
    const std::filesystem::path& path,
    const std::array<double, 3>& targetGridXyz,
    const cv::Vec3i& volumeShape,
    UmbilicusTargetGridAuthority authority,
    std::optional<double> targetVoxelSizeUm)
{
    UmbilicusFrameLoad result;
    const UmbilicusFile info = readUmbilicusFile(path);
    if (!info.metadataErrors.empty()) {
        // A file that garbles its own frame statement is refused outright: a
        // typo must never read as a legacy file that declared nothing.
        std::string error = "refusing umbilicus '" + path.string() +
                            "': malformed frame metadata:";
        for (const auto& entry : info.metadataErrors) {
            error += "\n  " + entry;
        }
        result.error = std::move(error);
        return result;
    }
    const UmbilicusFrameClaim claim = umbilicusFrameClaim(info);
    if (!claim.any()) {
        // Unstamped legacy file: keep the historical reading.
        result.umbilicus = Umbilicus::FromPoints(Umbilicus::PointsOf(info), volumeShape);
        return result;
    }
    const auto scale =
        deriveUmbilicusScale(info, targetGridXyz, targetVoxelSizeUm);
    // A contradiction refuses before anything else is weighed (see
    // decideUmbilicusLoadAction): the scale was derived from a statement the
    // named volume disproves. Under an Inferred grid that refusal can only
    // become warn+legacy — the grid is not the volume's own, so a mismatch
    // proves nothing about the file. The Apply arm below applies the same
    // caution in reverse: a fit against an inferred grid cannot confirm a
    // grid-derived rescale either, since a surface patch's bounding box is
    // not its frame and a partial surface can exactly mimic a downsampled
    // volume.
    const auto contradiction =
        umbilicusStampContradiction(info, targetGridXyz, targetVoxelSizeUm);
    const auto action = decideUmbilicusLoadAction(
        scale, claim, /*haveTargetGrid=*/true,
        /*stampContradicted=*/contradiction.has_value());
    if (action == UmbilicusLoadAction::Apply) {
        // An inferred grid cannot establish a grid-derived rescale, so one
        // is never applied on its say-so: warn and keep the legacy reading.
        // Only an explicit coordinate conversion — a stamped voxel size
        // against the target's own — survives, since it never depended on
        // the inferred grid at all.
        if (authority == UmbilicusTargetGridAuthority::Inferred &&
            scale->source != UmbilicusScaleSource::StampedVoxelSize) {
            result.warning =
                "umbilicus '" + path.string() +
                "' declares a frame implying a rescale (" +
                scale->description +
                "), but the working grid is inferred from surface bounds, "
                "which cannot confirm the rescale; using its points as-is.";
            result.umbilicus = Umbilicus::FromPoints(Umbilicus::PointsOf(info), volumeShape);
            return result;
        }
        // A scale read off the points means the stated frame was never
        // evaluated: point inference only runs when the stamped checks
        // produced no scale — a voxel-size-only stamp with no target voxel
        // size to compare it against. Rescaling on it would treat an
        // unchecked frame as checked, so under an authoritative grid the
        // file is refused instead of rescaled.
        if (authority == UmbilicusTargetGridAuthority::Authoritative &&
            scale->source == UmbilicusScaleSource::InferredFromGrid) {
            result.error =
                "refusing umbilicus '" + path.string() +
                "': its stamped voxelsize_um cannot be checked against "
                "this target (no target voxel size), so its points cannot "
                "be rescaled; provide the target voxel size or re-stamp "
                "the file.";
            return result;
        }
        auto points = Umbilicus::PointsOf(info);
        const float factor = static_cast<float>(scale->factor);
        for (auto& point : points) {
            point = point * factor;
        }
        result.umbilicus = Umbilicus::FromPoints(points, volumeShape);
        result.scaleDescription = scale->description;
        return result;
    }
    if (action == UmbilicusLoadAction::Refuse) {
        if (authority == UmbilicusTargetGridAuthority::Authoritative) {
            // The grid is the volume's own: a declared frame that fits
            // nothing about it means the file is wrong about itself, so
            // it is refused outright rather than read as though it had
            // declared nothing.
            const std::string reason =
                contradiction.has_value()
                    ? *contradiction
                    : "its declared frame does not fit the volume grid; "
                      "re-stamp it with volume_width, volume_height and "
                      "volume_slices.";
            result.error =
                "refusing umbilicus '" + path.string() + "': " + reason;
            return result;
        }
        // The grid is caller-inferred, not authoritative, so the mismatch
        // proves nothing about the file: a surface patch's bounding box is
        // not its frame. Warn loudly and keep the legacy reading rather
        // than breaking a working invocation over an unverifiable claim.
        result.warning =
            "umbilicus '" + path.string() +
            "' declares a frame that does not fit the working grid; using "
            "its points as-is. If the umbilicus was drawn on a different "
            "resolution of this data, re-stamp it with volume_width, "
            "volume_height and volume_slices.";
        result.umbilicus = Umbilicus::FromPoints(Umbilicus::PointsOf(info), volumeShape);
        return result;
    }
    // Unreachable while the claim is non-empty: decide() only answers Apply
    // or Refuse there. Kept so a future decide() change still fails safe.
    result.umbilicus = Umbilicus::FromPoints(Umbilicus::PointsOf(info), volumeShape);
    return result;
}

} // namespace vc::core::util
