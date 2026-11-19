#pragma once

#include "vc/lasagna/Dataset.hpp"
#include "vc/core/types/VolumePkg.hpp"

#include <vc_reader/registrations.hpp>

#include <array>
#include <filesystem>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

class Volume;
class VolumePkg;

namespace vc::lasagna
{

inline constexpr std::string_view kLasagnaVolumeManifestTagPrefix = "vc-lasagna-manifest:";
inline constexpr std::string_view kLasagnaVolumeGroupTagPrefix = "vc-lasagna-group:";

[[nodiscard]] std::vector<std::string> lasagnaProjectVolumeTags(
    const LasagnaChannelGroup& group,
    std::string_view manifestLocation);

// One volume of an open project, as the registration graph sees it: the volume
// its coordinates are expressed in, the scale those coordinates carry (the
// working level), and the shape at that level. A Lasagna group volume also
// carries the base shape its manifest declares.
struct LasagnaRegistrationVolume {
    std::string coordinateId;
    double coordinateScale = 1.0;
    std::array<std::size_t, 3> shapeZYX{};
    std::optional<std::array<std::size_t, 3>> lasagnaBaseShapeZYX;
};

// Registrations relating each Lasagna group volume to the volumes of the
// dataset it belongs to.
//
// Membership is decided by shape: a volume belongs when its shape is a level of
// the manifest's baseShapeZYX, through the dyadic comparison that tolerates the
// rounding a pyramid introduces. Every such volume is related, not only the one
// at the declared shape - a project holds the base volume and any number of
// derived levels, and they are all the same frame - and each mapping is that
// pair's power-of-two relation, never the literal ratio of their shapes,
// expressed between the two level-0 frames so that composing it the way
// VolumeTransforms::toVolume does lands on the target's working coordinates.
// The relations are all powers of two of one base, so the routes they add to
// the graph compose consistently rather than making it ambiguous.
[[nodiscard]] std::vector<vcr::catalog::Registration> lasagnaBaseShapeRegistrations(
    std::span<const LasagnaRegistrationVolume> volumes);

// The Lasagna registrations of an open project: every volume's coordinate
// frame, working scale and shape, with the base shape its manifest tag
// declares, run through lasagnaBaseShapeRegistrations. Manifests are read
// once per location and cached for the process (they are immutable).
// `remoteCacheRoot` is where a remote manifest is cached; empty uses the
// reader's default. A volume whose manifest cannot be read carries no base
// shape and so no edge.
[[nodiscard]] std::vector<vcr::catalog::Registration> lasagnaProjectRegistrations(
    const VolumePkg& pkg, const std::filesystem::path& remoteCacheRoot = {});

using PreparedLasagnaProjectVolume = VolumePkg::PreparedVolumeAttachment;

[[nodiscard]] std::vector<PreparedLasagnaProjectVolume> prepareLasagnaProjectVolumes(
    const LasagnaDataset& dataset,
    std::string manifestLocation = {});

}  // namespace vc::lasagna
