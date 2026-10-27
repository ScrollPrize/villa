#pragma once
#include <vc_reader/catalog.hpp>

#include "OpenDataManifest.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

class VolumePkg;
namespace vc::project { struct Entry; }
namespace vc::lasagna { struct LasagnaDatasetManifest; }

namespace vc3d::opendata {


struct OpenDataLasagnaInfo {
    std::string sampleId;
    std::string volumeId;
    std::size_t artifactIndex = 0;
    std::string artifactUrl;
    std::string modelId;
    int sourceCoordinateLevel = 0;
    bool lasagnaVersionPresent = false;
    std::optional<int> lasagnaVersion;
    bool sourceToBasePresent = false;
    std::optional<double> sourceToBase;
    std::optional<std::array<std::size_t, 3>> baseShapeZYX;
};

struct ResolvedOpenDataLasagna {
    std::filesystem::path manifestPath;
    std::string sourceManifestLocation;
    double workingToBaseScale = 1.0;
    std::string artifactUrl;
    bool manifestBacked = false;
};

enum class OpenDataLasagnaDatasetKind {
    Normal,
    FiberInference,
};

// Validates the catalog metadata against the authoritative inner manifest and
// classifies the dataset by its channel schema. Catalog creation_info is
// optional; when present, its values must agree with the inner manifest.
[[nodiscard]] OpenDataLasagnaDatasetKind validateOpenDataLasagnaManifest(
    const OpenDataLasagnaInfo& info,
    const vc::lasagna::LasagnaDatasetManifest& manifest);

// The catalog Lasagna artifact a dataset entry caches (the artifact_url of its
// remote marker); nullopt for other entries.
[[nodiscard]] std::optional<vcr::catalog::OpenDataVolumeView> lasagnaCatalogView(
    const VolumePkg& pkg, const vc::project::Entry& entry);

// Returns the public manifest URL for a catalogue-backed entry and the stored
// project location for an ordinary entry.
[[nodiscard]] std::string lasagnaSourceManifestLocation(
    const VolumePkg& pkg, const vc::project::Entry& entry);

// Resolve the single root manifest in a public catalogue artifact directory.
[[nodiscard]] std::string discoverOpenDataLasagnaManifestUrl(const std::string& artifactUrl);

[[nodiscard]] std::vector<OpenDataLasagnaInfo> lasagnaArtifacts(
    const std::string& sampleId,
    const vcr::catalog::OpenDataVolume& volume);

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
                          const vcr::catalog::OpenDataSample& sample,
                          const std::filesystem::path& remoteCacheRoot,
                          std::vector<std::string>* messages = nullptr,
                          const OpenDataResourceSelection* selection = nullptr);

[[nodiscard]] std::optional<ResolvedOpenDataLasagna> resolveLasagnaForVolume(
    const VolumePkg& pkg,
    const std::string& loadedVolumeId);

// The same for what the active volume shows in the catalog (nullopt: the
// project's manual selection), checked against its shape when given. No
// affine or cross-volume pairing is attempted.
[[nodiscard]] std::optional<ResolvedOpenDataLasagna> resolveLasagnaForView(
    const VolumePkg& pkg,
    const std::optional<vcr::catalog::OpenDataVolumeView>& view,
    const std::optional<std::array<int, 3>>& workingShapeZYX = std::nullopt);

} // namespace vc3d::opendata
