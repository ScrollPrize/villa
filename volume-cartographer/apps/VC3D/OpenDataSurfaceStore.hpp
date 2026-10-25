#pragma once
#include <vc_reader/catalog.hpp>
#include <vc_object_store/objects/ObjectStoreSession.hpp>
#include <vc_object_store/objects/VolumeTransforms.hpp>

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
    PublishedTransformed
};

struct OpenDataSegmentRepresentation {
    const vcr::catalog::OpenDataArtifact* artifact = nullptr;  // registered ones
    OpenDataSegmentRepresentationKind kind = OpenDataSegmentRepresentationKind::DerivedNative;
    bool canonicalSource = false;
    std::string coordinateVolumeId;
    std::string coordinateSpace;
    std::string representationId;
};

// The representations of a segment, at most one per sample volume. The source
// volume's is canonical: an explicit published representation targeting it
// wins, otherwise the authored artifact in native L0. Published ones win for
// every other volume. Other compatible volumes use transformed store views.
[[nodiscard]] std::vector<OpenDataSegmentRepresentation>
classifyOpenDataSegmentRepresentations(const vcr::catalog::OpenDataSample& sample,
                                       const vcr::catalog::OpenDataSegment& segment);

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

// A volume showing a catalog volume at pyramid level `level` (a source opened at
// a coarser base level) has its own id. Register it to the volume it shows
// (scale 2^-level) unless the store already relates the two, so objects bound
// at any level are seen at every other. The registration's UUID depends only on
// the pair, so replicas that both register it make one object, not two routes.
// Returns whether a registration was written.
bool registerVolumeLevel(vc::objects::ObjectStoreSession& session, const vc::objects::AttachmentIdentity& attachment,
                         const std::string& volume, const std::string& shown, int level);

// The mapping from volume `from`'s voxels to `to`'s, resolved by `transforms`
// (a store's, or a catalog's) with each volume that shows a catalog volume at
// some level (`fromShown`, `toShown`) related to it unless already related.
// nullopt when unrelated or ambiguous.
std::optional<vc::objects::VolumeAffine> volumeToVolume(vc::objects::VolumeTransforms transforms,
                                                        const std::string& from, const std::optional<CatalogVolume>& fromShown,
                                                        const std::string& to, const std::optional<CatalogVolume>& toShown);

struct SurfaceImportResult {
    int representations = 0;
    int added = 0;
    int updated = 0;
    std::vector<std::string> messages;
};
// Record the representations of the sample's segments (only `segmentIds` when
// given) in its store, creating the store. Re-importing updates the records;
// geometry already fetched stays.
SurfaceImportResult importSampleSurfaces(const std::filesystem::path& remoteCacheRoot,
                                         const vcr::catalog::OpenDataSample& sample,
                                         const std::optional<std::vector<std::string>>& segmentIds = std::nullopt);

enum class SurfaceState { Missing, Remote, Stored };
// The canonical representation of a segment: not imported, imported without
// geometry yet, or with geometry in the store.
[[nodiscard]] SurfaceState surfaceStateForSegment(const std::filesystem::path& remoteCacheRoot,
                                                  const vcr::catalog::OpenDataSample& sample,
                                                  const vcr::catalog::OpenDataSegment& segment);
[[nodiscard]] const char* surfaceStateName(SurfaceState state) noexcept;

// Fetch a catalog surface's registered tifxyz. Blocking; no-op when it has geometry.
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
