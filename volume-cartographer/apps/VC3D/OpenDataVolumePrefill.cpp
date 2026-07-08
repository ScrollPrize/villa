#include "OpenDataVolumePrefill.hpp"

#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/Logging.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <chrono>
#include <exception>
#include <fstream>
#include <system_error>
#include <utility>

namespace vc3d::opendata {
namespace {

constexpr const char* kMarkerVersion = "vc3d_open_data_prefill_v1";
constexpr std::size_t kProgressIntervalChunks = 32;

std::size_t chunkCountForGrid(const std::array<int, 3>& grid)
{
    return static_cast<std::size_t>(grid[0]) *
           static_cast<std::size_t>(grid[1]) *
           static_cast<std::size_t>(grid[2]);
}

OpenDataVolumePrefillMarkerInfo markerInfoForVolume(const Volume& volume, int level)
{
    OpenDataVolumePrefillMarkerInfo info;
    info.remoteUrl = volume.remoteUrl();
    info.volumeId = volume.id();
    info.level = level;
    info.shape = volume.levelShape(level);
    info.chunkShape = volume.chunkShape(level);
    info.chunkGridShape = volume.chunkGridShape(level);
    info.totalChunks = chunkCountForGrid(info.chunkGridShape);
    return info;
}

nlohmann::json markerJsonForInfo(const OpenDataVolumePrefillMarkerInfo& info)
{
    return nlohmann::json{
        {"version", kMarkerVersion},
        {"remote_url", info.remoteUrl},
        {"volume_id", info.volumeId},
        {"level", info.level},
        {"shape", info.shape},
        {"chunk_shape", info.chunkShape},
        {"chunk_grid_shape", info.chunkGridShape},
        {"chunk_count", info.totalChunks},
    };
}

bool markerMatchesJson(const nlohmann::json& marker,
                       const OpenDataVolumePrefillMarkerInfo& info)
{
    if (!marker.is_object()) {
        return false;
    }
    const auto expected = markerJsonForInfo(info);
    for (const char* key : {
             "version",
             "remote_url",
             "volume_id",
             "level",
             "shape",
             "chunk_shape",
             "chunk_grid_shape",
             "chunk_count",
         }) {
        const auto found = marker.find(key);
        if (found == marker.end() || *found != expected.at(key)) {
            return false;
        }
    }
    return true;
}

bool cancelled(const std::atomic<bool>* cancelFlag)
{
    return cancelFlag && cancelFlag->load(std::memory_order_acquire);
}

} // namespace

std::filesystem::path openDataVolumePrefillMarkerPath(
    const std::filesystem::path& cacheDir,
    int level)
{
    return cacheDir / (".vc_prefill_level_" + std::to_string(level) + ".json");
}

bool openDataVolumePrefillMarkerMatches(const std::filesystem::path& cacheDir,
                                        const OpenDataVolumePrefillMarkerInfo& info)
{
    if (cacheDir.empty()) {
        return false;
    }

    std::ifstream file(openDataVolumePrefillMarkerPath(cacheDir, info.level));
    if (!file) {
        return false;
    }

    try {
        const auto marker = nlohmann::json::parse(file, nullptr, false);
        return !marker.is_discarded() && markerMatchesJson(marker, info);
    } catch (...) {
        return false;
    }
}

bool openDataVolumePrefillMarkerMatches(const std::filesystem::path& cacheDir,
                                        const Volume& volume,
                                        int level)
{
    if (cacheDir.empty() || !volume.isRemote() || !volume.hasScaleLevel(level)) {
        return false;
    }
    return openDataVolumePrefillMarkerMatches(cacheDir, markerInfoForVolume(volume, level));
}

bool writeOpenDataVolumePrefillMarker(const std::filesystem::path& cacheDir,
                                      const OpenDataVolumePrefillMarkerInfo& info,
                                      std::string* errorOut)
{
    if (errorOut) {
        errorOut->clear();
    }
    if (cacheDir.empty()) {
        if (errorOut) {
            *errorOut = "remote cache directory is empty";
        }
        return false;
    }

    try {
        std::error_code ec;
        std::filesystem::create_directories(cacheDir, ec);
        if (ec) {
            if (errorOut) {
                *errorOut = ec.message();
            }
            return false;
        }

        auto marker = markerJsonForInfo(info);
        const auto now = std::chrono::system_clock::now().time_since_epoch();
        marker["completed_at_unix_ms"] =
            std::chrono::duration_cast<std::chrono::milliseconds>(now).count();

        const auto path = openDataVolumePrefillMarkerPath(cacheDir, info.level);
        const auto tmp = path.string() + ".tmp";
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (!out) {
                if (errorOut) {
                    *errorOut = "could not open marker temp file";
                }
                return false;
            }
            out << marker.dump(2) << '\n';
            if (!out) {
                if (errorOut) {
                    *errorOut = "could not write marker temp file";
                }
                return false;
            }
        }
        std::filesystem::rename(tmp, path, ec);
        if (ec) {
            std::filesystem::remove(path, ec);
            ec.clear();
            std::filesystem::rename(tmp, path, ec);
        }
        if (ec) {
            std::filesystem::remove(tmp, ec);
            if (errorOut) {
                *errorOut = ec.message();
            }
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        if (errorOut) {
            *errorOut = e.what();
        }
        return false;
    }
}

bool writeOpenDataVolumePrefillMarker(const std::filesystem::path& cacheDir,
                                      const Volume& volume,
                                      int level,
                                      std::size_t totalChunks,
                                      std::string* errorOut)
{
    auto info = markerInfoForVolume(volume, level);
    info.totalChunks = totalChunks;
    return writeOpenDataVolumePrefillMarker(cacheDir, info, errorOut);
}

OpenDataVolumePrefillResult prefillOpenDataVolumeLevel(
    std::shared_ptr<Volume> volume,
    int level,
    const std::atomic<bool>* cancelFlag,
    const OpenDataVolumePrefillProgressCallback& progressCallback)
{
    OpenDataVolumePrefillResult result;
    result.level = level;
    if (!volume) {
        result.message = "no volume";
        return result;
    }

    result.volumeId = volume->id();
    result.cacheDir = volume->remotePersistentCachePath();

    if (!volume->isRemote()) {
        result.message = "volume is local";
        return result;
    }
    if (result.cacheDir.empty()) {
        result.message = "remote persistent cache is not configured";
        return result;
    }
    if (!volume->hasScaleLevel(level)) {
        result.message = "remote volume does not have scale level " + std::to_string(level);
        return result;
    }
    if (openDataVolumePrefillMarkerMatches(result.cacheDir, *volume, level)) {
        result.status = OpenDataVolumePrefillResult::Status::Skipped;
        result.totalChunks = volume->chunkCount(level);
        result.resolvedChunks = result.totalChunks;
        result.message = "level already prefetched";
        return result;
    }
    if (cancelled(cancelFlag)) {
        result.status = OpenDataVolumePrefillResult::Status::Cancelled;
        result.message = "cancelled";
        return result;
    }

    try {
        const auto grid = volume->chunkGridShape(level);
        result.totalChunks = chunkCountForGrid(grid);
        if (result.totalChunks == 0) {
            result.status = OpenDataVolumePrefillResult::Status::Skipped;
            result.message = "level has no chunks";
            return result;
        }

        vc::render::ChunkCache::Options options;
        options.decodedByteCapacity = 64ULL * 1024ULL * 1024ULL;
        options.metadataEntryCapacity = std::max<std::size_t>(result.totalChunks, 1ULL << 12);
        options.maxConcurrentReads = 1;
        auto cache = volume->createChunkCache(std::move(options));
        if (!cache) {
            result.status = OpenDataVolumePrefillResult::Status::Failed;
            result.message = "could not create chunk cache";
            return result;
        }

        for (int iz = 0; iz < grid[0]; ++iz) {
            for (int iy = 0; iy < grid[1]; ++iy) {
                for (int ix = 0; ix < grid[2]; ++ix) {
                    if (cancelled(cancelFlag)) {
                        result.status = OpenDataVolumePrefillResult::Status::Cancelled;
                        result.message = "cancelled";
                        cache->waitForPersistentWrites();
                        return result;
                    }

                    const auto chunk = cache->getChunkBlocking(level, iz, iy, ix);
                    ++result.resolvedChunks;
                    switch (chunk.status) {
                    case vc::render::ChunkStatus::Data:
                        ++result.dataChunks;
                        break;
                    case vc::render::ChunkStatus::Missing:
                    case vc::render::ChunkStatus::AllFill:
                        ++result.emptyChunks;
                        break;
                    case vc::render::ChunkStatus::Error:
                        ++result.errorChunks;
                        Logger()->warn(
                            "Open-data volume prefill error for {} level {} chunk {}/{}/{}: {}",
                            result.volumeId,
                            level,
                            iz,
                            iy,
                            ix,
                            chunk.error);
                        break;
                    case vc::render::ChunkStatus::MissQueued:
                        ++result.errorChunks;
                        break;
                    }

                    if (progressCallback &&
                        (result.resolvedChunks == result.totalChunks ||
                         result.resolvedChunks % kProgressIntervalChunks == 0)) {
                        progressCallback(result.resolvedChunks, result.totalChunks);
                    }
                }
            }
        }

        cache->waitForPersistentWrites();
        if (result.errorChunks > 0) {
            result.status = OpenDataVolumePrefillResult::Status::Failed;
            result.message = std::to_string(result.errorChunks) + " chunk(s) failed";
            return result;
        }

        std::string markerError;
        if (!writeOpenDataVolumePrefillMarker(
                result.cacheDir, *volume, level, result.totalChunks, &markerError)) {
            result.status = OpenDataVolumePrefillResult::Status::Failed;
            result.message = "could not write prefill marker: " + markerError;
            return result;
        }

        result.status = OpenDataVolumePrefillResult::Status::Completed;
        result.message = "completed";
        return result;
    } catch (const std::exception& e) {
        result.status = OpenDataVolumePrefillResult::Status::Failed;
        result.message = e.what();
    } catch (...) {
        result.status = OpenDataVolumePrefillResult::Status::Failed;
        result.message = "unknown error";
    }
    return result;
}

} // namespace vc3d::opendata
