#pragma once

#include <atomic>
#include <cstddef>
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
    int level = kOpenDataVolumePrefillLevel;
    int physicalLevel = kOpenDataVolumePrefillLevel;
    std::size_t totalChunks = 0;
    std::size_t resolvedChunks = 0;
    std::size_t dataChunks = 0;
    std::size_t emptyChunks = 0;
    std::size_t errorChunks = 0;
    std::size_t fetchedChunks = 0;  // read from the remote; the rest were already cached
    std::string message;
};

using OpenDataVolumePrefillProgressCallback =
    std::function<void(std::size_t resolvedChunks, std::size_t totalChunks)>;

// Read every chunk of working level `level` of a remote volume once, so that
// the volume's disk cache holds the whole level. Chunks already cached are
// read from disk and not fetched again.
OpenDataVolumePrefillResult prefillOpenDataVolumeLevel(
    std::shared_ptr<Volume> volume,
    int level = kOpenDataVolumePrefillLevel,
    const std::atomic<bool>* cancelFlag = nullptr,
    const OpenDataVolumePrefillProgressCallback& progressCallback = {});

} // namespace vc3d::opendata
