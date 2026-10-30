#include "OpenDataVolumePrefill.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/util/Logging.hpp"

#include <vc_reader/zarr.hpp>

#include <atomic>
#include <exception>
#include <mutex>

namespace vc3d::opendata {
namespace {
constexpr std::size_t kProgressIntervalChunks = 32;
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
    result.physicalLevel = volume->baseScaleLevel() + level;
    if (!volume->isRemote()) {
        result.message = "volume is local";
        return result;
    }
    if (volume->remoteCacheRoot().empty()) {
        result.message = "remote persistent cache is not configured";
        return result;
    }
    const auto cancelled = [cancelFlag] {
        return cancelFlag && cancelFlag->load(std::memory_order_acquire);
    };

    try {
        // No memory cache: fetched chunks go to the volume's disk cache only.
        auto opened = vcr::zarr::openPyramidLevel(
            volume->sourceLocation(), result.physicalLevel, volume->sourceOpenOptions());
        auto& array = *opened.array;
        const auto grid = array.chunk_grid();
        result.totalChunks = 1;
        for (const auto n : grid)
            result.totalChunks *= static_cast<std::size_t>(n);

        std::mutex resultMutex;
        array.parallel_for(result.totalChunks, [&](std::size_t i) {
            if (cancelled())
                return;
            vcr::Index idx(grid.size());
            for (std::size_t d = grid.size(); d-- > 0;) {
                idx[d] = static_cast<std::int64_t>(i % static_cast<std::size_t>(grid[d]));
                i /= static_cast<std::size_t>(grid[d]);
            }
            bool stored = false;
            std::string error;
            try {
                stored = array.chunk(idx) != nullptr;
            } catch (const std::exception& e) {
                error = e.what();
            }
            std::lock_guard lock(resultMutex);
            ++result.resolvedChunks;
            if (!error.empty()) {
                ++result.errorChunks;
                Logger()->warn("Open-data volume prefill error for {} level {}: {}",
                               result.volumeId, level, error);
            } else if (stored) {
                ++result.dataChunks;
            } else {
                ++result.emptyChunks;
            }
            if (progressCallback &&
                (result.resolvedChunks == result.totalChunks ||
                 result.resolvedChunks % kProgressIntervalChunks == 0)) {
                progressCallback(result.resolvedChunks, result.totalChunks);
            }
        });
        array.flush();
        result.fetchedChunks = array.stats().remote_fetches.load();

        if (cancelled() && result.resolvedChunks < result.totalChunks) {
            result.status = OpenDataVolumePrefillResult::Status::Cancelled;
            result.message = "cancelled";
        } else if (result.errorChunks > 0) {
            result.status = OpenDataVolumePrefillResult::Status::Failed;
            result.message = std::to_string(result.errorChunks) + " chunk(s) failed";
        } else if (array.stats().write_errors.load() > 0) {
            result.status = OpenDataVolumePrefillResult::Status::Failed;
            result.message = "cache write failed: " + array.last_write_error();
        } else {
            result.status = OpenDataVolumePrefillResult::Status::Completed;
            result.message = "completed";
        }
    } catch (const std::exception& e) {
        result.status = OpenDataVolumePrefillResult::Status::Failed;
        result.message = e.what();
    }
    return result;
}

} // namespace vc3d::opendata
