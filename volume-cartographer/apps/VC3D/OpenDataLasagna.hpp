#pragma once

#include "OpenDataManifest.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

class VolumePkg;

namespace vc3d::opendata {

inline constexpr std::string_view kLasagnaArtifactType = "lasagna";
inline constexpr std::string_view kOpenDataLasagnaEntryTag = "vc-open-data-lasagna";
inline constexpr std::string_view kOpenDataLasagnaArtifactTagPrefix =
    "vc-open-data-lasagna-artifact:";
inline constexpr std::string_view kOpenDataLasagnaModelTagPrefix =
    "vc-open-data-lasagna-model-id:";

struct OpenDataLasagnaInfo {
    std::string sampleId;
    std::string volumeId;
    std::string artifactUrl;
    std::string modelId;
    bool lasagnaVersionPresent = false;
    std::optional<int> lasagnaVersion;
    bool sourceToBasePresent = false;
    std::optional<double> sourceToBase;
    std::optional<std::array<std::size_t, 3>> baseShapeZYX;
};

struct ResolvedOpenDataLasagna {
    std::filesystem::path manifestPath;
    double workingToBaseScale = 1.0;
    std::string coordinateSpace;
    std::string artifactUrl;
    bool manifestBacked = false;
};

[[nodiscard]] std::vector<OpenDataLasagnaInfo> lasagnaArtifacts(
    const std::string& sampleId,
    const OpenDataVolume& volume);

[[nodiscard]] std::filesystem::path lasagnaCacheDir(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataLasagnaInfo& info);

// Download and validate the small dataset manifest and Zarr descriptors. Data
// chunks remain lazy and are persisted by the core read-through store.
[[nodiscard]] std::filesystem::path prepareOpenDataLasagna(
    const OpenDataLasagnaInfo& info,
    const std::filesystem::path& remoteCacheRoot,
    std::string* errorOut = nullptr);

// selection optionally restricts attached volumes and representations;
// nullptr attaches everything.
int attachOpenDataLasagna(VolumePkg& pkg,
                          const OpenDataSample& sample,
                          const std::filesystem::path& remoteCacheRoot,
                          std::vector<std::string>* messages = nullptr,
                          const OpenDataResourceSelection* selection = nullptr);

[[nodiscard]] std::optional<ResolvedOpenDataLasagna> resolveLasagnaForVolume(
    const VolumePkg& pkg,
    const std::string& loadedVolumeId);

// Public for non-GUI consumers and deterministic pairing tests. The tags must
// describe one catalog coordinate identity; no affine or cross-volume pairing
// is attempted.
[[nodiscard]] std::optional<ResolvedOpenDataLasagna> resolveLasagnaForCoordinateTags(
    const VolumePkg& pkg,
    const std::vector<std::string>& volumeTags);

[[nodiscard]] std::optional<ResolvedOpenDataLasagna> resolveLasagnaForCoordinateShape(
    const VolumePkg& pkg,
    const std::vector<std::string>& volumeTags,
    const std::array<int, 3>& workingShapeZYX);

} // namespace vc3d::opendata
