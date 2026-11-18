#pragma once

#include "vc/lasagna/Dataset.hpp"
#include "vc/core/types/VolumePkg.hpp"

#include <memory>
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

using PreparedLasagnaProjectVolume = VolumePkg::PreparedVolumeAttachment;

[[nodiscard]] std::vector<PreparedLasagnaProjectVolume> prepareLasagnaProjectVolumes(
    const LasagnaDataset& dataset,
    std::string manifestLocation = {});

}  // namespace vc::lasagna
