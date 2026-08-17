#include "vc/core/util/ScrollUmbilicus.hpp"

#include <algorithm>
#include <optional>
#include <limits>
#include <cmath>
#include <array>
#include <exception>
#include <system_error>

#include "vc/core/types/VolumePkg.hpp"

namespace fs = std::filesystem;

namespace {

// A tifxyz surface directory carries all three coordinate planes; a directory of
// such surfaces carries none of its own. Checking the payload rather than
// meta.json keeps this from matching directories that happen to hold a file by
// that name for unrelated reasons.
bool isTifxyzSegmentDir(const fs::path& dir)
{
    std::error_code ec;
    for (const char* plane : {"x.tif", "y.tif", "z.tif"}) {
        if (!fs::exists(dir / plane, ec) || ec) {
            return false;
        }
    }
    return true;
}


fs::path canonicalize(const fs::path& path)
{
    std::error_code ec;
    auto canonical = fs::weakly_canonical(path, ec);
    if (ec) {
        return path.lexically_normal();
    }
    return canonical;
}

std::string joinPaths(const std::vector<fs::path>& paths)
{
    std::string joined;
    for (const auto& path : paths) {
        if (!joined.empty()) {
            joined += ", ";
        }
        joined += path.string();
    }
    return joined;
}

// Malformed frame metadata is a hard failure here rather than in every
// consumer: this function is the one place that answers "which umbilicus, in
// which frame", so a typo must not be allowed to look like an unstamped file.
std::string describeMetadataErrors(const fs::path& path,
                                   const std::vector<std::string>& errors)
{
    std::string message =
        "the umbilicus file " + path.string() + " declares malformed metadata: ";
    for (std::size_t i = 0; i < errors.size(); ++i) {
        if (i != 0) {
            message += "; ";
        }
        message += errors[i];
    }
    return message;
}

} // namespace

namespace vc::core::util {

ScrollUmbilicusResolution resolveScrollUmbilicus(const VolumePkg& pkg)
{
    ScrollUmbilicusResolution resolution;

    // The raw configured location, not umbilicusPath(), decides whether the
    // field is set: umbilicusPath() blanks remote locations, and a blank there
    // must not be mistaken for "nothing configured" and trigger a search.
    if (const auto configured = pkg.umbilicus(); !configured.empty()) {
        const auto declared = pkg.umbilicusPath();
        if (declared.empty()) {
            resolution.error =
                "the project's \"umbilicus\" field is set to " + configured +
                ", a remote or otherwise unsupported location that cannot be "
                "used as an umbilicus file";
            return resolution;
        }
        std::error_code ec;
        if (!fs::exists(declared, ec) || ec) {
            resolution.error =
                "the project's \"umbilicus\" field points at " +
                declared.string() + ", which does not exist";
            return resolution;
        }
        try {
            auto info = Umbilicus::LoadFileInfo(declared);
            if (!info.metadataErrors.empty()) {
                resolution.error =
                    describeMetadataErrors(declared, info.metadataErrors);
                return resolution;
            }
            resolution.info = std::move(info);
            resolution.path = declared;
        } catch (const std::exception& e) {
            resolution.error =
                "the project's \"umbilicus\" field points at " +
                declared.string() + ", which could not be loaded: " + e.what();
        }
        return resolution;
    }

    std::vector<fs::path> roots;
    const auto addRoot = [&roots](const fs::path& root) {
        if (root.empty()) {
            return;
        }
        if (std::find(roots.begin(), roots.end(), root) != roots.end()) {
            return;
        }
        roots.push_back(root);
    };

    addRoot(pkg.path().empty() ? fs::path(pkg.getVolpkgDirectory())
                               : pkg.path().parent_path());
    for (const auto& segment : pkg.availableSegmentPaths()) {
        // The entry may be the segments directory itself, whose parent is the
        // volpkg, or a single segment inside it, whose grandparent is. Only the
        // latter earns the extra level: taking the grandparent of a segments
        // directory would search whatever holds the volpkg, where an unrelated
        // umbilicus could win or force a false ambiguity.
        addRoot(segment.parent_path());
        if (isTifxyzSegmentDir(segment)) {
            addRoot(segment.parent_path().parent_path());
        }
    }

    std::vector<fs::path> hits;
    std::vector<fs::path> canonicalHits;
    for (const auto& root : roots) {
        for (const char* name : {"umbilicus.json", "estimated_umbilicus.json"}) {
            const fs::path candidate = root / name;
            std::error_code ec;
            if (!fs::exists(candidate, ec) || ec) {
                continue;
            }
            const auto canonical = canonicalize(candidate);
            if (std::find(canonicalHits.begin(), canonicalHits.end(), canonical) ==
                canonicalHits.end()) {
                canonicalHits.push_back(canonical);
                hits.push_back(candidate);
            }
            break;
        }
    }

    if (hits.empty()) {
        resolution.error =
            "no umbilicus.json or estimated_umbilicus.json found; searched: " +
            (roots.empty() ? std::string("(no candidate roots)")
                           : joinPaths(roots));
        return resolution;
    }

    if (hits.size() > 1) {
        resolution.ambiguous = hits;
        resolution.error =
            "found several umbilicus files (" + joinPaths(hits) +
            "); set the project's \"umbilicus\" field to pick one";
        return resolution;
    }

    try {
        auto info = Umbilicus::LoadFileInfo(hits.front());
        if (!info.metadataErrors.empty()) {
            resolution.error =
                describeMetadataErrors(hits.front(), info.metadataErrors);
            return resolution;
        }
        resolution.info = std::move(info);
        resolution.path = hits.front();
    } catch (const std::exception& e) {
        resolution.error = "failed to load " + hits.front().string() + ": " +
                           e.what();
    }
    return resolution;
}

std::optional<UmbilicusScale> deriveUmbilicusScale(
    const UmbilicusFileInfo& info,
    const std::array<double, 3>& targetGridXyz,
    std::optional<double> targetVoxelSizeUm)
{
    const bool haveTarget = targetGridXyz[0] > 0.0 && targetGridXyz[1] > 0.0 &&
                            targetGridXyz[2] > 0.0;

    const bool haveStampedDims = info.volumeWidth && info.volumeHeight &&
                                 info.volumeSlices && *info.volumeWidth > 0 &&
                                 *info.volumeHeight > 0 && *info.volumeSlices > 0;
    if (haveStampedDims && haveTarget) {
        const std::array<double, 3> ratios{
            targetGridXyz[0] / *info.volumeWidth,
            targetGridXyz[1] / *info.volumeHeight,
            targetGridXyz[2] / *info.volumeSlices};
        const double lo = std::min({ratios[0], ratios[1], ratios[2]});
        const double hi = std::max({ratios[0], ratios[1], ratios[2]});
        if (lo > 0.0 && hi <= lo * 1.02) {
            UmbilicusScale scale;
            scale.factor = (ratios[0] + ratios[1] + ratios[2]) / 3.0;
            scale.source = UmbilicusScaleSource::StampedDimensions;
            scale.description = "stamped " + std::to_string(*info.volumeWidth) + "x" +
                                std::to_string(*info.volumeHeight) + "x" +
                                std::to_string(*info.volumeSlices) + " grid";
            return scale;
        }
        // Axis ratios that disagree mean this is not a rescale of the target
        // grid at all, so neither of the weaker readings below applies either.
        return std::nullopt;
    }

    if (info.voxelsizeUm && targetVoxelSizeUm && *targetVoxelSizeUm > 0.0) {
        const double ratio = *info.voxelsizeUm / *targetVoxelSizeUm;
        if (std::isfinite(ratio) && ratio > 0.0) {
            UmbilicusScale scale;
            scale.factor = ratio;
            scale.source = UmbilicusScaleSource::StampedVoxelSize;
            scale.description = "stamped " + std::to_string(*info.voxelsizeUm) +
                                " um voxels";
            return scale;
        }
    }

    if (!haveTarget || info.controlPoints.empty()) {
        return std::nullopt;
    }

    // Nothing stated: read the grid off the points. An umbilicus runs the length
    // of the scroll, so the right grid is the one it nearly fills and still fits
    // inside. Candidates are a factor of two apart, so a grid this well covered
    // leaves the next coarser one under half covered and the match is unique.
    constexpr double kMinZCoverage = 0.6;
    constexpr int kMaxDownsampleSteps = 5;
    std::array<double, 3> lo{std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::infinity()};
    std::array<double, 3> hi{-std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity()};
    for (const auto& point : info.controlPoints) {
        for (int axis = 0; axis < 3; ++axis) {
            lo[axis] = std::min(lo[axis], static_cast<double>(point[axis]));
            hi[axis] = std::max(hi[axis], static_cast<double>(point[axis]));
        }
    }

    std::optional<UmbilicusScale> match;
    for (int step = 0; step <= kMaxDownsampleSteps; ++step) {
        const double candidate = std::pow(2.0, step);
        bool fits = lo[0] >= 0.0 && lo[1] >= 0.0 && lo[2] >= 0.0;
        for (int axis = 0; axis < 3 && fits; ++axis) {
            fits = hi[axis] <= targetGridXyz[axis] / candidate;
        }
        if (!fits) {
            continue;
        }
        const double gridZ = targetGridXyz[2] / candidate;
        const double coverage = gridZ > 0.0 ? (hi[2] - lo[2]) / gridZ : 0.0;
        if (coverage < kMinZCoverage) {
            continue;
        }
        if (match) {
            return std::nullopt;  // more than one grid fits: say nothing
        }
        UmbilicusScale scale;
        scale.factor = candidate;
        scale.source = UmbilicusScaleSource::InferredFromGrid;
        scale.description = "inferred from the volume grid";
        match = scale;
    }
    return match;
}

} // namespace vc::core::util
