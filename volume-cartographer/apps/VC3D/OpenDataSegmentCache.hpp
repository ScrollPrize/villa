#pragma once

#include "OpenDataManifest.hpp"

#include <filesystem>
#include <functional>
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

struct OpenDataInkDetectionEntry {
    std::string label;
    std::string sampleId;
    std::string segmentId;
    std::string segmentLongId;
    std::string artifactType;
    std::string sourceUrl;
    std::filesystem::path localPath;
};

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

[[nodiscard]] OpenDataSegmentCacheState cacheStateForSegment(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample,
    const OpenDataSegment& segment);

[[nodiscard]] bool isOpenDataCatalogSegmentDirectory(
    const std::filesystem::path& segmentDir);

[[nodiscard]] std::vector<OpenDataInkDetectionEntry> cachedInkDetectionsForSegmentDirectory(
    const std::filesystem::path& segmentDir);

[[nodiscard]] std::filesystem::path defaultEditableCopyPathForCatalogSegment(
    const std::filesystem::path& catalogSegmentDir,
    const std::filesystem::path& remoteCacheRoot);

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
