#pragma once

#include <cstddef>
#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>
#include "vc_reader/affine.hpp"
#include "vc_reader/registrations.hpp"

namespace vcr::catalog {

inline constexpr std::string_view kDefaultManifestUrl =
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json";

// creation.derived_from: what a record was made from ({"type": "volume", "id": ...}).
struct OpenDataReference {
    std::string type;
    std::string id;
};

// A record's creation block (creation, or an artifact's creation_info).
struct OpenDataCreation {
    std::string date;     // as published (ISO 8601)
    std::string process;
    std::optional<OpenDataReference> derivedFrom;
};

// An axis-aligned box in a volume's voxels (x, y, z).
struct OpenDataBounds {
    std::array<double, 3> low{0, 0, 0};
    std::array<double, 3> high{0, 0, 0};
};

struct OpenDataAccessRoot {
    std::string type;
    std::string url;
    std::string usage;
    nlohmann::json properties = nlohmann::json::object();
    nlohmann::json raw = nlohmann::json::object();
};

struct OpenDataOrigin {
    std::string path;
    // copy_info.source_path: where the published copy was copied from.
    std::string copySourcePath;
    std::vector<OpenDataAccessRoot> accessRoots;
    nlohmann::json properties = nlohmann::json::object();
    nlohmann::json raw = nlohmann::json::object();
};

struct OpenDataArtifact {
    std::string type;
    std::vector<OpenDataOrigin> origins;
    nlohmann::json parameters = nlohmann::json::object();
    nlohmann::json properties = nlohmann::json::object();
    nlohmann::json creationInfo = nlohmann::json::object();
    nlohmann::json raw = nlohmann::json::object();

    // Exact typed parameters used by coordinate-bearing catalog artifacts.
    // levelParameterPresent distinguishes a missing value from malformed or
    // out-of-range manifest data (both leave sourceCoordinateLevel empty).
    bool levelParameterPresent = false;
    std::optional<int> sourceCoordinateLevel;
    std::optional<std::string> targetVolumeId;
    std::optional<std::string> modelId;
    bool lasagnaVersionPresent = false;
    std::optional<int> lasagnaVersion;
    bool sourceToBasePresent = false;
    std::optional<double> sourceToBase;

    // Typed parameters and properties of rendered and labelled artifacts.
    std::optional<int> renderLevel;
    std::optional<double> downscale;
    std::string format;
    std::string compositeMode;
    std::string labelSet;
    std::optional<int> width;   // properties.width/height (OBJ and layer artifacts)
    std::optional<int> height;
    // ink-labels properties: label arrays, superseded label sets, how the
    // labels reached this artifact's volume and how much of it they cover.
    std::vector<std::string> arrays;
    std::vector<std::string> supersedes;
    std::string registration;
    std::string annotatedOn;
    std::optional<double> mappingCoverage;
    OpenDataCreation creation;  // from creation_info

    // The <type> of a published path under predictions/<type>/ (surfaces,
    // lasagna, fibers, ink-3d); empty elsewhere.
    std::string predictionType;

    // Preferred public origin selected from origins/access_roots.
    std::string sourcePath;
    std::string resolvedUrl;
    std::string accessUsage;

    [[nodiscard]] bool hasResolvedUrl() const noexcept;
    // The artifact's URL: the resolved public URL, else its source path,
    // without trailing slashes. Empty when it has no public origin.
    [[nodiscard]] std::string url() const;
};

std::vector<OpenDataArtifact> parseCatalogArtifacts(const nlohmann::json& entries);

struct OpenDataVolumeTransform {
    std::string toVolumeId;
    Affine matrix = Affine::identity();
    std::string derivationPath;
    nlohmann::json raw = nlohmann::json::object();
};

struct OpenDataVolumeTransformGroup {
    std::string fromVolumeId;
    std::vector<OpenDataVolumeTransform> transforms;
    nlohmann::json raw = nlohmann::json::object();
};

struct OpenDataScan {
    std::string id;
    std::string longId;
    std::string suffix;
    std::string createdAt;
    OpenDataCreation creation;
    std::optional<double> pixelSizeUm;
    std::optional<double> energyKeV;
    std::optional<double> detectorDistanceMm;
    // Acquisition, from creation.metadata: facility location and the
    // facility's scan name (ESRF scanRadix, DLS scan_name).
    std::string location;
    std::string name;
    nlohmann::json properties = nlohmann::json::object();
    std::vector<OpenDataArtifact> artifacts;
    nlohmann::json raw = nlohmann::json::object();
};

// properties.superseded_by: the volume that replaces this one.
struct OpenDataSupersession {
    std::string volumeId;
    std::string date;
    std::string reason;
};

struct OpenDataLicense {
    std::string name;
    std::string url;
};

// properties.z_direction_is_top_to_bottom / left_handed_coordinates: how the
// volume's axes lie against the scroll (slice z = 0 is the top; the xy plane
// is mirrored). Unset where the catalog does not say.
struct OpenDataOrientation {
    std::optional<bool> zTopToBottom;
    std::optional<bool> leftHandedCoordinates;
};

struct OpenDataVolume {
    std::string id;
    std::string longId;
    std::string scanId;
    std::string suffix;
    OpenDataCreation creation;
    OpenDataOrientation orientation;
    std::string textSide;
    std::optional<OpenDataSupersession> supersededBy;
    std::optional<OpenDataLicense> license;
    int renderLevel = 0;
    std::optional<bool> renderSegments;
    // Intensity windowing of the export (creation.metadata).
    std::optional<double> windowU16Min, windowU16Max;
    std::optional<std::array<std::size_t, 3>> shapeZYX;
    std::optional<double> pixelSizeUm;
    std::optional<double> energyKeV;
    std::optional<double> detectorDistanceMm;
    std::string dataFormat;
    std::string createdAt;
    nlohmann::json properties = nlohmann::json::object();
    std::vector<OpenDataArtifact> artifacts;
    nlohmann::json raw = nlohmann::json::object();
};

// properties.volume_coverage[volume]: where a segment lies in a volume.
struct OpenDataVolumeCoverage {
    std::string volumeId;
    OpenDataBounds bbox;       // bbox_transformed, in that volume's voxels
    double overlapRatio = 0;   // 0: the segment does not reach the volume
    std::string computedAt;
};

// The original tifxyz recorded at segment creation (creation.metadata).
struct OpenDataTifxyzInfo {
    std::array<double, 2> scale{0, 0};
    std::optional<OpenDataBounds> bbox;
    std::optional<std::array<int64_t, 2>> tiffDimensions;  // as published
};

struct OpenDataSegment {
    std::string id;
    std::string longId;
    std::string suffix;
    std::string originalVolumeId;
    std::optional<int> width;
    std::optional<int> height;
    std::string createdAt;
    OpenDataCreation creation;
    // Voxels of the original volume per authored coordinate unit.
    double originalVolumeDownscale = 1;
    std::vector<OpenDataVolumeCoverage> volumeCoverage;
    nlohmann::json render;  // properties.render, as published (null when absent)
    std::optional<OpenDataTifxyzInfo> originalTifxyz;
    nlohmann::json properties = nlohmann::json::object();
    std::vector<OpenDataArtifact> artifacts;
    nlohmann::json raw = nlohmann::json::object();

    [[nodiscard]] bool hasInkDetection() const;
    [[nodiscard]] bool hasLayersZarr() const;
    // The volume the segment was authored in: creation.derived_from when it
    // names a volume, else original_volume_id.
    [[nodiscard]] std::string sourceVolumeId() const;
    // The segment's identity across catalog revisions: long_id, else id.
    [[nodiscard]] std::string stableId() const;
    [[nodiscard]] const OpenDataVolumeCoverage* coverage(std::string_view volumeId) const noexcept;
};

// A tifxyz representation of a segment that can be opened as a surface.
struct OpenDataSurface {
    const OpenDataArtifact* artifact = nullptr;
    // Authored: the segment's own tifxyz, in its source volume's downscaled
    // voxels. Published: a tifxyz-transformed onto volumeId.
    bool authored = false;
    // The segment's own surface (in its source volume), not a published copy
    // in another volume.
    bool canonical = false;
    std::string volumeId;       // the catalog volume its coordinates are in
    double unitsPerVoxel = 1;   // coordinate units per voxel of volumeId
    // Stable across catalog revisions: the segment's stable id and which
    // representation ("<lineage>-authored", "<lineage>-published-<volume>").
    std::string id;
};

// A published per-segment image (ink detection), fetched beside the surface.
struct OpenDataSegmentImage {
    const OpenDataArtifact* artifact = nullptr;
    std::string url;
    std::string fileName;  // e.g. "ink-detection.jpg"
    std::string label;
};
// The segment's ink detection images: one file per ink artifact with an image
// URL, preferring JPEG renders when the segment publishes any.
[[nodiscard]] std::vector<OpenDataSegmentImage> inkDetectionImages(const OpenDataSegment&);


struct OpenDataSample {
    std::string id;
    std::string type;
    std::string description;
    nlohmann::json publicAccessMappings;  // properties.public_access_mappings (null when absent)
    nlohmann::json properties = nlohmann::json::object();
    std::vector<OpenDataArtifact> artifacts;
    std::vector<OpenDataScan> scans;
    std::vector<OpenDataVolume> volumes;
    std::vector<OpenDataSegment> segments;
    std::vector<OpenDataVolumeTransformGroup> volumeTransforms;
    nlohmann::json raw = nlohmann::json::object();

    [[nodiscard]] std::size_t scanCount() const noexcept;
    [[nodiscard]] std::size_t volumeCount() const noexcept;
    [[nodiscard]] std::size_t segmentCount() const noexcept;
    // Segments with at least one surface (surfaces() is not empty).
    [[nodiscard]] std::size_t surfaceSegmentCount() const;
    [[nodiscard]] std::size_t inkDetectionSegmentCount() const;
    [[nodiscard]] const OpenDataVolume* findVolume(std::string_view id) const noexcept;
    // The segment's surfaces, the canonical one first: a tifxyz-transformed
    // published onto its source volume, else its authored tifxyz (preferring
    // tifxyz-flattened, then tifxyz, then tifxyz-normalized) when its downscale
    // is a power of two; then its tifxyz-transformed onto the sample's other
    // volumes. Empty when its source volume is not in the sample.
    [[nodiscard]] std::vector<OpenDataSurface> surfaces(const OpenDataSegment&) const;
};

enum class OpenDataRepresentationKind {
    NormalGrids,
    Lasagna,
    Prediction,
};

struct OpenDataRepresentationRef {
    std::size_t volumeIndex = 0;
    std::size_t artifactIndex = 0;
    OpenDataRepresentationKind kind = OpenDataRepresentationKind::Prediction;
};

// Manifest-only view of volume-derived representations. Raw source volume
// artifacts are intentionally excluded.
[[nodiscard]] std::vector<OpenDataRepresentationRef> derivedRepresentations(
    const OpenDataSample& sample);
[[nodiscard]] std::string_view representationKindName(
    OpenDataRepresentationKind kind) noexcept;

// Per-artifact classification shared by derivedRepresentations() (its single
// authority) and by the resource-selection filters below. Returns nullopt for a
// raw source-volume artifact (i.e. not a derived representation).
[[nodiscard]] std::optional<OpenDataRepresentationKind>
classifyDerivedRepresentation(const OpenDataArtifact& artifact);

struct OpenDataModel {
    std::string id;
    std::string longId;
    std::string suffix;
    OpenDataCreation creation;
    std::string task;
    std::string architecture;
    nlohmann::json properties = nlohmann::json::object();
    nlohmann::json raw = nlohmann::json::object();
};

// The id of a catalog volume shown at a pyramid level: the volume's own id at
// level 0, "<id>@L<level>" otherwise. Registration graphs built from the
// catalog relate every level to its volume (RegistrationGraph::addSample).
[[nodiscard]] std::string levelVolumeId(std::string_view volumeId, int level);

// What an opened volume shows: a catalog volume of a sample at a pyramid level.
struct OpenDataVolumeView {
    std::string sampleId;
    std::string volumeId;
    int level = 0;
    // The volume's own zarr (at a base level), not a dataset derived from it
    // (prediction, normal grids, Lasagna) on its grid.
    bool source = true;
    const OpenDataArtifact* artifact = nullptr;
    [[nodiscard]] std::string frameId() const { return levelVolumeId(volumeId, level); }
};

struct OpenDataManifest {
    std::string manifestUrl;
    std::vector<OpenDataSample> samples;
    std::vector<OpenDataModel> models;
    nlohmann::json raw = nlohmann::json::object();

    [[nodiscard]] const OpenDataSample* findSample(std::string_view id) const noexcept;
    // The catalog volume a remote zarr shows when opened at pyramid
    // `baseLevel`: a volume's preferred zarr, or a derived dataset (its
    // parameters.level added to the base level). URLs compare without trailing
    // slashes, as resolved or as published.
    [[nodiscard]] std::optional<OpenDataVolumeView> volumeAt(std::string_view url, int baseLevel = 0) const;
};

// The manifest in a cache file (loadCatalog's cacheFile), parsed once per file
// version and shared; nullptr when the file is missing or invalid.
[[nodiscard]] std::shared_ptr<const OpenDataManifest> cachedManifest(const std::filesystem::path& cacheFile);

[[nodiscard]] OpenDataManifest parseOpenDataManifest(
    std::string_view jsonText,
    std::string manifestUrl = std::string(kDefaultManifestUrl));

[[nodiscard]] OpenDataManifest loadOpenDataManifestFile(
    const std::filesystem::path& manifestPath,
    std::string manifestUrl = {});



// The https:// location of an s3:// URL (public channels by their published
// rewrite, other buckets by region: s3+REGION://); other URLs are unchanged.
[[nodiscard]] std::string resolveOpenDataUrl(std::string url);
// The s3:// form of a URL in a public channel (or already s3://), else nullopt.
[[nodiscard]] std::optional<std::string> openDataS3Url(std::string url);
[[nodiscard]] std::string joinOpenDataUrl(std::string root, std::string path);
// A catalog id as one cache path component: characters other than
// [A-Za-z0-9._-] become '_' and leading '.'/'_' are dropped ("unnamed" if empty).
[[nodiscard]] std::string safePathComponent(std::string value);

[[nodiscard]] const OpenDataArtifact* findArtifact(
    const std::vector<OpenDataArtifact>& artifacts,
    std::string_view type) noexcept;

// A Zarr array or pyramid: its type names Zarr (ome-zarr,
// surface-prediction-zarr, ...) or its URL path ends in .zarr.
[[nodiscard]] bool isZarrArtifact(const OpenDataArtifact& artifact);

[[nodiscard]] const OpenDataArtifact* preferredVolumeArtifact(
    const OpenDataVolume& volume) noexcept;

[[nodiscard]] const OpenDataArtifact* preferredPhotoArtifact(
    const OpenDataSample& sample) noexcept;


struct CatalogLoadOptions {
    std::filesystem::path cacheFile;
    bool refresh = false;
    double maxAgeSeconds = 3600;
    double timeoutSeconds = 60;
};
struct CatalogLoadResult {
    OpenDataManifest manifest;
    bool fromCache = false;
    std::string warning;
};
CatalogLoadResult loadCatalog(const std::string& source = std::string(kDefaultManifestUrl),
                              const CatalogLoadOptions& options = {});
} // namespace vcr::catalog
