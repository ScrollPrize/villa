#pragma once
#include <vc_reader/catalog.hpp>

#include "OpenDataManifest.hpp"
#include "vc/core/types/VolumePkg.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace vc::lasagna { struct LasagnaDatasetManifest; }

namespace vc3d::opendata {

enum class OpenDataLasagnaDatasetKind {
    Normal,
    FiberInference,
};

// The volume's attachable Lasagna artifacts, by index: classified as Lasagna,
// with a URL and parameters.level; one per published dataset.
[[nodiscard]] std::vector<std::size_t> lasagnaArtifacts(const vcr::catalog::OpenDataVolume& volume);

// Validates the catalog metadata against the authoritative inner manifest and
// classifies the dataset by its channel schema. Catalog creation_info is
// optional; when present, its values must agree with the inner manifest.
[[nodiscard]] OpenDataLasagnaDatasetKind validateOpenDataLasagnaManifest(
    const vcr::catalog::OpenDataVolume& volume,
    const vcr::catalog::OpenDataArtifact& artifact,
    const vc::lasagna::LasagnaDatasetManifest& manifest);

// Returns the public manifest URL for a catalogue-backed entry and the stored
// project location for an ordinary entry.
[[nodiscard]] std::string lasagnaSourceManifestLocation(
    const VolumePkg& pkg, const vc::project::Entry& entry);

// A sample's Lasagna, cached and validated against the catalog with its
// volumes opened, ready to attach. Preparing does the network work and
// touches no project.
struct OpenDataLasagnaPreparation {
    struct Dataset {
        std::string manifest;  // the cached manifest: the entry's location
        bool fiberInference = false;
        std::vector<VolumePkg::PreparedVolumeAttachment> volumes;
    };
    std::string sampleId;
    std::vector<std::string> artifactUrls;  // every artifact prepared or tried
    std::vector<Dataset> datasets;
    std::vector<std::string> messages;      // artifacts skipped or failed
};

// selection optionally restricts the volumes and representations prepared;
// nullptr prepares everything. Data chunks stay lazy (the core read-through
// store persists them).
[[nodiscard]] OpenDataLasagnaPreparation prepareOpenDataLasagna(
    const vcr::catalog::OpenDataSample& sample,
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataResourceSelection* selection = nullptr);

// Attaches the prepared datasets and removes the project's other cached copies
// of the same artifacts (superseded or failed downloads). Returns the number
// attached; failures are appended to messages.
int attachOpenDataLasagna(VolumePkg& pkg,
                          const OpenDataLasagnaPreparation& preparation,
                          std::vector<std::string>* messages = nullptr);

struct ResolvedOpenDataLasagna {
    std::filesystem::path manifestPath;
    std::string sourceManifestLocation;
    double workingToBaseScale = 1.0;
    std::string artifactUrl;
    bool manifestBacked = false;
};

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
