#pragma once

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vc3d::opendata {

struct CoordinateIdentity {
    std::string coordinateSpace;
    std::string sourcePath;
    int sourceCoordinateLevel = 0;
    std::uint64_t sourceCoordinateScaleFactor = 1;
    double sourceOriginalResolution = 0.0;
};

inline std::optional<CoordinateIdentity> coordinateIdentityFromTags(
    const std::vector<std::string>& tags)
{
    constexpr std::string_view spacePrefix = "vc-open-data-coordinate-space:";
    constexpr std::string_view levelPrefix =
        "vc-open-data-source-coordinate-level:";
    constexpr std::string_view sourcePathPrefix =
        "vc-open-data-source-path:";
    constexpr std::string_view originalResolutionPrefix =
        "vc-open-data-source-original-resolution:";
    std::optional<std::string> space;
    std::optional<std::string> sourcePath;
    std::optional<int> level;
    std::optional<double> originalResolution;
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
        } else if (tag.rfind(sourcePathPrefix, 0) == 0) {
            sourcePath = tag.substr(sourcePathPrefix.size());
        } else if (tag.rfind(originalResolutionPrefix, 0) == 0) {
            const auto value = tag.substr(originalResolutionPrefix.size());
            try {
                std::size_t consumed = 0;
                const double parsed = std::stod(value, &consumed);
                if (consumed == value.size() && std::isfinite(parsed) && parsed > 0.0)
                    originalResolution = parsed;
            } catch (...) {
            }
        }
    }
    if (!space || space->empty() || !level)
        return std::nullopt;
    return CoordinateIdentity{
        *space,
        sourcePath.value_or(std::string{}),
        *level,
        std::uint64_t{1} << *level,
        originalResolution.value_or(0.0),
    };
}

inline std::optional<CoordinateIdentity> coordinateIdentityForVolume(
    const VolumePkg& pkg,
    const std::string& loadedVolumeId)
{
    auto identity = coordinateIdentityFromTags(pkg.volumeTags(loadedVolumeId));
    if (!identity)
        return std::nullopt;

    const auto volume = pkg.volume(loadedVolumeId);
    if (identity->sourcePath.empty() && volume) {
        identity->sourcePath = volume->isRemote()
            ? volume->remoteUrl()
            : volume->path().string();
    }
    if (!(identity->sourceOriginalResolution > 0.0) && volume) {
        try {
            const double logicalResolution = volume->voxelSize();
            if (std::isfinite(logicalResolution) && logicalResolution > 0.0) {
                identity->sourceOriginalResolution =
                    logicalResolution /
                    static_cast<double>(identity->sourceCoordinateScaleFactor);
            }
        } catch (...) {
        }
    }
    if (identity->sourcePath.empty() ||
        !(identity->sourceOriginalResolution > 0.0)) {
        return std::nullopt;
    }
    return identity;
}

inline utils::Json coordinateIdentityJson(
    const std::optional<CoordinateIdentity>& identity)
{
    auto metadata = utils::Json::object();
    if (!identity)
        return metadata;
    metadata["vc_open_data_coordinate_space"] = identity->coordinateSpace;
    metadata["vc_open_data_source_path"] = identity->sourcePath;
    metadata["vc_open_data_source_coordinate_level"] =
        identity->sourceCoordinateLevel;
    metadata["vc_open_data_source_coordinate_scale_factor"] =
        identity->sourceCoordinateScaleFactor;
    metadata["vc_open_data_source_original_resolution"] =
        identity->sourceOriginalResolution;
    return metadata;
}

inline void copyCoordinateIdentityToJson(
    utils::Json& target,
    const std::optional<CoordinateIdentity>& identity)
{
    if (!identity)
        return;
    if (target.is_null() || !target.is_object())
        target = utils::Json::object();
    target.update(coordinateIdentityJson(identity));
}

inline void copyCoordinateIdentityToSurface(
    QuadSurface& surface,
    const std::optional<CoordinateIdentity>& identity)
{
    copyCoordinateIdentityToJson(surface.meta, identity);
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
