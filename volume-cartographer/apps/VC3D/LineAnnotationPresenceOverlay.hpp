#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Pure pieces of the line annotation dialog's fiber presence overlay: the
// user settings it persists, and the lookups the controller runs to turn the
// selected fiber dataset into a volume the panes can draw.
namespace vc3d::line_annotation
{

struct PresenceOverlaySettings {
    bool enabled = false;
    // Blend opacity in [0,1] at full presence.
    double opacity = 0.6;
    // "#rrggbb"; the pane colormap is tint:rrggbb.
    std::string colorHex = "#ff00ff";
    // Overlay window low: presence below it is not drawn at all, and the
    // value-weighted alpha ramps up from it.
    int threshold = 16;
    // Advanced mode draws any project volume (volumeId) instead of the
    // selected fiber dataset's presence channel; the simple mode is default.
    bool advanced = false;
    std::string volumeId;
};

inline constexpr std::string_view kLasagnaManifestTagPrefix = "vc-lasagna-manifest:";
inline constexpr std::string_view kLasagnaGroupTagPrefix = "vc-lasagna-group:";

// A project volume together with its tags, as VolumePkg exposes them.
struct TaggedVolumeId {
    std::string id;
    std::vector<std::string> tags;
};

// The id of the attached volume that holds `groupName` of one of the manifest
// locations in `manifestCandidates` (the stored selection first, then any
// alternate spelling such as the catalogue URL). Candidates are tried in
// order, so a volume of the stored location wins over an alias.
inline std::optional<std::string> findLasagnaGroupVolumeId(
    const std::vector<TaggedVolumeId>& volumes,
    const std::vector<std::string>& manifestCandidates,
    std::string_view groupName)
{
    const std::string groupTag =
        std::string(kLasagnaGroupTagPrefix) + std::string(groupName);
    for (const auto& manifest : manifestCandidates) {
        if (manifest.empty()) {
            continue;
        }
        const std::string manifestTag =
            std::string(kLasagnaManifestTagPrefix) + manifest;
        for (const auto& volume : volumes) {
            const bool hasManifest =
                std::find(volume.tags.begin(), volume.tags.end(), manifestTag) !=
                volume.tags.end();
            const bool hasGroup =
                std::find(volume.tags.begin(), volume.tags.end(), groupTag) !=
                volume.tags.end();
            if (hasManifest && hasGroup) {
                return volume.id;
            }
        }
    }
    return std::nullopt;
}

// How many pyramid levels the presence volume has to be rebased by so that its
// level 0 lands on the active volume's grid. `fiberBaseToVolumeScale` carries
// fiber-manifest base coordinates into the active volume (volume = base *
// scale), so 1 means no rebase, 0.5 one level, 0.25 two. Nullopt when the
// scale is not a whole number of downsampling levels (an upsampled view, or a
// non-dyadic ratio).
inline std::optional<int> presenceRebaseLevel(double fiberBaseToVolumeScale)
{
    if (!std::isfinite(fiberBaseToVolumeScale) || fiberBaseToVolumeScale <= 0.0 ||
        fiberBaseToVolumeScale > 1.0) {
        return std::nullopt;
    }
    const double levels = -std::log2(fiberBaseToVolumeScale);
    const double rounded = std::round(levels);
    if (std::abs(levels - rounded) > 1e-6 || rounded > 30.0) {
        return std::nullopt;
    }
    return static_cast<int>(rounded);
}

// One stored level of an overlay pyramid, as the Volume reports it.
struct StoredPyramidLevel {
    int level = 0;
    std::array<int, 3> shapeZYX{};
    // Outer storage chunk (the shard for sharded arrays): exporters may pad a
    // level out to whole storage chunks.
    std::array<int, 3> storageChunkShapeZYX{};
};

// Every rebase level k (view level 0 = source level k) at which the pyramid
// fits a view whose level 0 is the active grid. Candidate k ranges over the
// source levels; for each, the view's finest retained level is the first
// stored level at or coarser than k, and it must have the shape the active
// grid implies there: ceil(active / 2^(storedLevel - k)) exactly, or padded
// above it by less than one storage chunk. This is the fit a synthesized
// frame cannot give: a /3 export of a 20812-wide scroll stores 2602 columns,
// and 2602 x 8 = 20816 is not the scroll, but ceil(20812 / 8) is 2602. A full
// /0../4 pyramid fits a scroll @L1 at k = 1 (its /1 is then level 0), a
// /3,/4 export fits @L4 at k = 4. The result is empty when nothing fits (a
// single-level array attached as level 0, an unrelated volume) and has more
// than one entry only for degenerate tiny shapes.
inline std::vector<int> rebaseLevelsFittingPyramid(
    const std::array<std::size_t, 3>& activeShapeZYX,
    const std::vector<StoredPyramidLevel>& storedLevels)
{
    std::vector<int> fits;
    int lastLevel = -1;
    for (const auto& stored : storedLevels) {
        lastLevel = std::max(lastLevel, stored.level);
    }
    for (int rebase = 0; rebase <= lastLevel; ++rebase) {
        const StoredPyramidLevel* finest = nullptr;
        for (const auto& stored : storedLevels) {
            if (stored.level >= rebase && (!finest || stored.level < finest->level)) {
                finest = &stored;
            }
        }
        if (!finest) {
            continue;
        }
        const int viewLevel = finest->level - rebase;
        bool ok = true;
        for (std::size_t axis = 0; axis < 3 && ok; ++axis) {
            const std::size_t divisor = std::size_t{1} << viewLevel;
            const std::size_t expected =
                (activeShapeZYX[axis] + divisor - 1) / divisor;
            const long long actual = finest->shapeZYX[axis];
            const long long padding = std::max(1, finest->storageChunkShapeZYX[axis]);
            ok = actual >= static_cast<long long>(expected) &&
                 actual - static_cast<long long>(expected) < padding;
        }
        if (ok) {
            fits.push_back(rebase);
        }
    }
    return fits;
}

// Whether every stored level of a pyramid has the shape the exact frame it
// was published against implies at that level index: ceil(frame / 2^level),
// or padded above it by less than one storage chunk. Used when a manifest
// records the frame: the dyadic matcher then gives the rebase level (it
// tolerates the one-voxel count/inclusive-maximum difference between frame
// conventions), and this check confirms the stored arrays really are that
// frame's pyramid (a single level attached as level 0 is not).
inline bool storedLevelsConsistentWithFrame(
    const std::array<std::size_t, 3>& frameShapeZYX,
    const std::vector<StoredPyramidLevel>& storedLevels)
{
    if (storedLevels.empty()) {
        return false;
    }
    for (const auto& stored : storedLevels) {
        if (stored.level < 0 || stored.level > 30) {
            return false;
        }
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const std::size_t divisor = std::size_t{1} << stored.level;
            const std::size_t expected = (frameShapeZYX[axis] + divisor - 1) / divisor;
            const long long actual = stored.shapeZYX[axis];
            const long long padding = std::max(1, stored.storageChunkShapeZYX[axis]);
            if (actual < static_cast<long long>(expected) ||
                actual - static_cast<long long>(expected) >= padding) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace vc3d::line_annotation
