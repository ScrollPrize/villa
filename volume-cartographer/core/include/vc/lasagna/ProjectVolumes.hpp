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

inline constexpr std::string_view kLasagnaDerivedVolumeTagPrefix = "vc-lasagna-derived:";
inline constexpr std::string_view kLasagnaGroupTagPrefix = "vc-lasagna-group:";
inline constexpr std::string_view kLasagnaChannelTagPrefix = "vc-lasagna-channel:";
inline constexpr std::string_view kLasagnaSpacingTagPrefix = "vc-lasagna-spacing:";

struct PreparedLasagnaProjectVolume {
    std::string location;
    std::vector<std::string> tags;
    std::shared_ptr<Volume> volume;
};

[[nodiscard]] std::vector<PreparedLasagnaProjectVolume> prepareLasagnaProjectVolumes(
    const LasagnaDataset& dataset,
    std::string manifestLocation = {});

// Recreate missing in-memory prepared 3D volumes for manifests already stored
// in a project. Returns non-fatal per-manifest diagnostics.
[[nodiscard]] std::vector<std::string> reconcileLasagnaProjectVolumes(VolumePkg& package);

}  // namespace vc::lasagna
