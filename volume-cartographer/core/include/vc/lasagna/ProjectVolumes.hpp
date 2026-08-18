#pragma once

#include "vc/lasagna/Dataset.hpp"

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

struct PreparedLasagnaProjectVolume {
    std::string location;
    std::vector<std::string> tags;
    std::shared_ptr<Volume> volume;
};

[[nodiscard]] std::vector<PreparedLasagnaProjectVolume> prepareLasagnaProjectVolumes(
    const LasagnaDataset& dataset,
    std::string manifestLocation = {});

}  // namespace vc::lasagna
