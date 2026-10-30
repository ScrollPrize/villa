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

// Whether cacheDir holds the completion marker prefillOpenDataVolumeLevel
// writes for this volume's level.
[[nodiscard]] bool openDataVolumePrefillMarkerMatches(
    const std::filesystem::path& cacheDir,
    const Volume& volume,
    int level = kOpenDataVolumePrefillLevel);

OpenDataVolumePrefillResult prefillOpenDataVolumeLevel(
    std::shared_ptr<Volume> volume,
    int level = kOpenDataVolumePrefillLevel,
    const std::atomic<bool>* cancelFlag = nullptr,
    const OpenDataVolumePrefillProgressCallback& progressCallback = {});

} // namespace vc3d::opendata
