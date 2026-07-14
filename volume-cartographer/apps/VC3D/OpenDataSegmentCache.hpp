#pragma once

#include "OpenDataManifest.hpp"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

class VolumePkg;

namespace vc3d::opendata {

enum class OpenDataSegmentCacheState {
    Missing,
    Current,
    Incomplete,
    Stale,
    Orphaned
};

struct OpenDataSampleDownloadProgress {
    int totalSegments = 0;
    int completedSegments = 0;
    int failedSegments = 0;
    int totalFiles = 0;
    int completedFiles = 0;
    int activeWorkers = 0;
    int totalWorkers = 0;
    std::string segmentId;
    std::string fileName;
    std::string status;
};

using OpenDataSampleProgressCallback =
    std::function<void(const OpenDataSampleDownloadProgress&)>;

struct OpenDataSegmentCacheReconcileResult {
    int supportedTifxyzSegments = 0;
    int cachedTifxyzSegments = 0;
    int attachedSegmentEntries = 0;
    int skippedTifxyzSegments = 0;
    int failedTifxyzSegments = 0;
    int transformedTifxyzSegments = 0;
    int failedTransformedTifxyzSegments = 0;
    std::vector<std::string> messages;
};

struct OpenDataSegmentMaterializationResult {
    bool success = false;
    bool alreadyMaterialized = false;
    int materializedSegments = 0;
    int failedSegments = 0;
    std::string message;
};

using OpenDataSegmentMaterializationProgress =
    std::function<void(int completed, int total,
                       const std::filesystem::path& segmentDir,
                       const std::string& status)>;

struct OpenDataInkDetectionEntry {
    std::string label;
    std::string sampleId;
    std::string segmentId;
    std::string segmentLongId;
    std::string artifactType;
    std::string sourceUrl;
    std::filesystem::path localPath;
};

enum class OpenDataSegmentRepresentationKind {
    Authored,
    DerivedNative,
    PublishedTransformed
};

struct OpenDataSegmentRepresentation {
    const OpenDataArtifact* artifact = nullptr;
    OpenDataSegmentRepresentationKind kind =
        OpenDataSegmentRepresentationKind::Authored;
    bool canonicalSource = false;
    std::string coordinateVolumeId;
    int sourceCoordinateLevel = 0;
    std::string coordinateSpace;
    std::string representationId;
};

// Select at most one representation per coordinate volume. The manifest's
// source volume is canonical: an explicit published representation targeting
// that volume wins, otherwise one authored artifact is converted to native L0.
// Explicit published representations win for every other target as well.
[[nodiscard]] std::vector<OpenDataSegmentRepresentation>
classifyOpenDataSegmentRepresentations(const OpenDataSample& sample,
                                       const OpenDataSegment& segment);

[[nodiscard]] std::filesystem::path openDataSegmentRepresentationCacheRoot(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample,
    const OpenDataSegmentRepresentation& representation);

[[nodiscard]] std::filesystem::path openDataSegmentRepresentationCacheDirectory(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample,
    const OpenDataSegment& segment,
    const OpenDataSegmentRepresentation& representation);

[[nodiscard]] std::filesystem::path openDataCanonicalSegmentCacheDirectory(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample,
    const OpenDataSegment& segment);

[[nodiscard]] const char* cacheStateName(OpenDataSegmentCacheState state) noexcept;

[[nodiscard]] std::filesystem::path openDataSegmentCacheRoot(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample,
    const std::string& sourceVolumeId);

[[nodiscard]] std::filesystem::path openDataSegmentCacheDirectory(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample,
    const OpenDataSegment& segment);

[[nodiscard]] std::filesystem::path openDataTransformedSegmentCacheRoot(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample,
    const std::string& sourceVolumeId,
    const std::string& targetVolumeId);

[[nodiscard]] std::filesystem::path openDataTransformedSegmentCacheDirectory(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample,
    const OpenDataSegment& segment,
    const std::string& targetVolumeId);

[[nodiscard]] std::filesystem::path openDataEditableSegmentRoot(
    const std::filesystem::path& remoteCacheRoot,
    const std::string& sampleId);

[[nodiscard]] std::filesystem::path openDataPatchesRoot(
    const std::filesystem::path& remoteCacheRoot,
    const std::string& sampleId);

[[nodiscard]] std::size_t manualOpenDataSegmentCount(
    const std::filesystem::path& remoteCacheRoot,
    const std::string& sampleId);

[[nodiscard]] OpenDataSegmentCacheState cacheStateForSegment(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample,
    const OpenDataSegment& segment);

[[nodiscard]] bool isOpenDataCatalogSegmentDirectory(
    const std::filesystem::path& segmentDir);

// Open-data placeholders contain complete display metadata and a persisted
// materialization recipe, but defer x/y/z TIFF download or generation until
// the surface is activated.
[[nodiscard]] bool isOpenDataSegmentPlaceholder(
    const std::filesystem::path& segmentDir);

[[nodiscard]] OpenDataSegmentMaterializationResult materializeOpenDataSegment(
    const std::filesystem::path& segmentDir);

[[nodiscard]] OpenDataSegmentMaterializationResult
materializeOpenDataSegmentFolder(
    const std::filesystem::path& segmentsRoot,
    const OpenDataSegmentMaterializationProgress& progressCallback = {});

[[nodiscard]] std::vector<OpenDataInkDetectionEntry> cachedInkDetectionsForSegmentDirectory(
    const std::filesystem::path& segmentDir);

[[nodiscard]] std::filesystem::path defaultEditableCopyPathForCatalogSegment(
    const std::filesystem::path& catalogSegmentDir,
    const std::filesystem::path& activeSegmentsRoot);

void copyCatalogSegmentToEditableDirectory(
    const std::filesystem::path& catalogSegmentDir,
    const std::filesystem::path& editableSegmentDir);

OpenDataSegmentCacheReconcileResult reconcileOpenDataSampleSegments(
    VolumePkg& pkg,
    const OpenDataSample& sample,
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSampleProgressCallback& progressCallback = {},
    bool forceRefresh = false);

OpenDataSegmentCacheReconcileResult attachExistingOpenDataSegmentCaches(
    VolumePkg& pkg,
    const OpenDataSample& sample,
    const std::filesystem::path& remoteCacheRoot);

} // namespace vc3d::opendata
