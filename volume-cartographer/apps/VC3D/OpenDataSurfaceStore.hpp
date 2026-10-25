#pragma once

// Open Data segments as surfaces of one internal object store per sample
// (<remote cache>/open_data/stores/<sample>.vcobj). Only this importer writes
// it. Every segment representation is a surface record bound to its catalog
// volume at L0 coordinates; its geometry stays remote (a registered tifxyz
// URL, or a volume matrix applied to another representation) until the surface
// is first opened. VC3D lists the store read-only beside the user's store.

#include "OpenDataManifest.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/core/matx.hpp>

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace vc3d::opendata {

enum class OpenDataSegmentRepresentationKind {
    // The authored tifxyz of the source volume, in its downscaled voxels.
    DerivedNative,
    // A published tifxyz in a target volume's L0 voxels.
    PublishedTransformed,
    // The source representation moved into a target volume by the sample's
    // volume matrix (no published one for that volume).
    Generated
};

struct OpenDataSegmentRepresentation {
    const OpenDataArtifact* artifact = nullptr;  // registered ones
    OpenDataSegmentRepresentationKind kind = OpenDataSegmentRepresentationKind::DerivedNative;
    bool canonicalSource = false;
    std::string coordinateVolumeId;
    std::string coordinateSpace;
    std::string representationId;
    // Generated: the representation it is computed from and the matrix.
    std::string sourceRepresentationId;
    std::optional<cv::Matx44d> matrix;
};

// The representations of a segment, at most one per sample volume. The source
// volume's is canonical: an explicit published representation targeting it
// wins, otherwise the authored artifact in native L0. Published ones win for
// every other volume; the rest of the volumes with a sample matrix from the
// source volume get a generated one.
[[nodiscard]] std::vector<OpenDataSegmentRepresentation>
classifyOpenDataSegmentRepresentations(const OpenDataSample& sample,
                                       const OpenDataSegment& segment);

[[nodiscard]] std::filesystem::path surfaceStorePath(const std::filesystem::path& remoteCacheRoot,
                                                     const std::string& sampleId);
// Deterministic across machines: a UUIDv5 of the representation id.
[[nodiscard]] std::string surfaceUuidForRepresentation(const std::string& representationId);

// The catalog volume an open Open Data volume shows, from its volume tags.
struct CatalogVolume {
    std::string sampleId;
    std::string volumeId;
    int level = 0;
};
[[nodiscard]] std::optional<CatalogVolume> catalogVolumeFromTags(const std::vector<std::string>& tags);

struct SurfaceImportResult {
    int representations = 0;
    int added = 0;
    int updated = 0;
    std::vector<std::string> messages;
};
// Record the representations of the sample's segments (only `segmentIds` when
// given) in its store, creating the store. Re-importing updates the records;
// geometry already fetched or generated stays.
SurfaceImportResult importSampleSurfaces(const std::filesystem::path& remoteCacheRoot,
                                         const OpenDataSample& sample,
                                         const std::optional<std::vector<std::string>>& segmentIds = std::nullopt);

enum class SurfaceState { Missing, Remote, Stored };
// The canonical representation of a segment: not imported, imported without
// geometry yet, or with geometry in the store.
[[nodiscard]] SurfaceState surfaceStateForSegment(const std::filesystem::path& remoteCacheRoot,
                                                  const OpenDataSample& sample,
                                                  const OpenDataSegment& segment);
[[nodiscard]] const char* surfaceStateName(SurfaceState state) noexcept;

// Give a catalog surface its geometry: fetch the registered tifxyz, or compute
// a generated one from its source (fetched first when needed). Blocking; no-op
// when it has geometry.
void materializeSurface(const std::filesystem::path& store, const std::string& uuid);
[[nodiscard]] bool surfaceHasGeometry(const std::filesystem::path& store, const std::string& uuid);

// The published ink detection images of a catalog surface (record meta), kept
// as files in its artifact directory.
struct InkDetection {
    std::string label;
    std::string artifactType;
    std::string url;
    std::filesystem::path file;  // in the artifact directory; may not exist yet
};
[[nodiscard]] std::vector<InkDetection> inkDetections(const nlohmann::json& surfaceMeta,
                                                      const std::filesystem::path& artifactDirectory);
// Download the image into its file unless it is there. Blocking.
void fetchInkDetection(const InkDetection& detection);

} // namespace vc3d::opendata
