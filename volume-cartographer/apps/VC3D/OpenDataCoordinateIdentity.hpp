#pragma once

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vc3d::opendata {

struct CoordinateIdentity {
    std::string coordinateSpace;
    int sourceCoordinateLevel = 0;
};

inline std::optional<CoordinateIdentity> coordinateIdentityFromTags(
    const std::vector<std::string>& tags)
{
    constexpr std::string_view spacePrefix = "vc-open-data-coordinate-space:";
    constexpr std::string_view levelPrefix =
        "vc-open-data-source-coordinate-level:";
    std::optional<std::string> space;
    std::optional<int> level;
    for (const auto& tag : tags) {
        if (tag.rfind(spacePrefix, 0) == 0) {
            space = tag.substr(spacePrefix.size());
        } else if (tag.rfind(levelPrefix, 0) == 0) {
            const auto value = tag.substr(levelPrefix.size());
            try {
                std::size_t consumed = 0;
                const int parsed = std::stoi(value, &consumed);
                if (consumed == value.size() && parsed >= 0 && parsed <= 5)
                    level = parsed;
            } catch (...) {
            }
        }
    }
    if (!space || space->empty() || !level)
        return std::nullopt;
    return CoordinateIdentity{*space, *level};
}

inline std::optional<CoordinateIdentity> coordinateIdentityForVolume(
    const VolumePkg& pkg,
    const std::string& loadedVolumeId)
{
    return coordinateIdentityFromTags(pkg.volumeTags(loadedVolumeId));
}

inline void copyCoordinateIdentityToSurface(
    QuadSurface& surface,
    const std::optional<CoordinateIdentity>& identity)
{
    if (!identity)
        return;
    if (surface.meta.is_null() || !surface.meta.is_object())
        surface.meta = utils::Json::object();
    surface.meta["vc_open_data_coordinate_space"] = identity->coordinateSpace;
    surface.meta["vc_open_data_source_coordinate_level"] =
        identity->sourceCoordinateLevel;
}

inline void copyVolumeCoordinateIdentityToSurface(
    QuadSurface& surface,
    const VolumePkg& pkg,
    const std::string& loadedVolumeId)
{
    copyCoordinateIdentityToSurface(
        surface, coordinateIdentityForVolume(pkg, loadedVolumeId));
}

} // namespace vc3d::opendata
