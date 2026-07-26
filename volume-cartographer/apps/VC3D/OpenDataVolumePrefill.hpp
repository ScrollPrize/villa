#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

class Volume;

namespace vc3d::opendata {

inline constexpr int kOpenDataVolumePrefillLevel = 5;

struct OpenDataVolumePrefillResult {
    enum class Status {
        Completed,
        Skipped,
        Cancelled,
        Failed,
    };

    Status status = Status::Skipped;
    std::string volumeId;
    std::filesystem::path cacheDir;
    int level = kOpenDataVolumePrefillLevel;
    int physicalLevel = kOpenDataVolumePrefillLevel;
    std::size_t totalChunks = 0;
    std::size_t resolvedChunks = 0;
    std::size_t dataChunks = 0;
    std::size_t emptyChunks = 0;
    std::size_t errorChunks = 0;
    std::string message;
};

using OpenDataVolumePrefillProgressCallback =
    std::function<void(std::size_t resolvedChunks, std::size_t totalChunks)>;

struct OpenDataVolumePrefillMarkerInfo {
    std::string remoteUrl;
    std::string remoteLocator;
    std::string volumeId;
    int level = kOpenDataVolumePrefillLevel;
    int physicalLevel = kOpenDataVolumePrefillLevel;
    std::array<int, 3> shape{};
    std::array<int, 3> chunkShape{};
    std::array<int, 3> chunkGridShape{};
    std::size_t totalChunks = 0;
};

[[nodiscard]] std::filesystem::path openDataVolumePrefillMarkerPath(
    const std::filesystem::path& cacheDir,
    int level = kOpenDataVolumePrefillLevel);

[[nodiscard]] bool openDataVolumePrefillMarkerMatches(
    const std::filesystem::path& cacheDir,
    const OpenDataVolumePrefillMarkerInfo& info);

[[nodiscard]] bool openDataVolumePrefillMarkerMatches(
    const std::filesystem::path& cacheDir,
    const Volume& volume,
    int level = kOpenDataVolumePrefillLevel);

bool writeOpenDataVolumePrefillMarker(
    const std::filesystem::path& cacheDir,
    const OpenDataVolumePrefillMarkerInfo& info,
    std::string* errorOut = nullptr);

bool writeOpenDataVolumePrefillMarker(
    const std::filesystem::path& cacheDir,
    const Volume& volume,
    int level,
    std::size_t totalChunks,
    std::string* errorOut = nullptr);

OpenDataVolumePrefillResult prefillOpenDataVolumeLevel(
    std::shared_ptr<Volume> volume,
    int level = kOpenDataVolumePrefillLevel,
    const std::atomic<bool>* cancelFlag = nullptr,
    const OpenDataVolumePrefillProgressCallback& progressCallback = {});

} // namespace vc3d::opendata
