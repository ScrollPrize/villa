// The vc_reader render pipeline (OpenedVolumes + vcr::grid), CPU or CUDA.

#include "bench_session.hpp"

#include "CState.hpp"
#include "OpenedVolumes.hpp"
#include "VCSettings.hpp"

#include "vc/core/types/Volume.hpp"

#include <QSettings>

#include <vc_reader/cache.hpp>
#include <vc_reader/cuda_available.hpp>
#include <vc_reader/array.hpp>
#include <vc_reader/zarr.hpp>

#include <algorithm>
#include <cstdio>

#include <stdexcept>

namespace vcbench::pipeline {

void writeSettings(Fixture& fixture, const std::string& iniPath)
{
    using namespace vc3d::settings;
    QSettings settings(QString::fromStdString(iniPath), QSettings::IniFormat);
    settings.setValue(perf::RENDER_CUDA, fixture.backend() == "cuda");
    settings.setValue(perf::RENDER_GPU_CACHE_GB, fixture.integer("gpu_cache_gb", 8));
    settings.sync();
}

void initProcess(Fixture& fixture)
{
    if (fixture.backend() == "cuda" && !vcr::cuda::available())
        throw std::runtime_error("--backend cuda: no CUDA device or a build without CUDA");
}

std::unique_ptr<CState> makeState(Fixture& fixture)
{
    return std::make_unique<CState>(nullptr, nullptr, std::size_t(fixture.num("cache_gb", 16)) << 30);
}

std::shared_ptr<Volume> openVolume(const VolumeSource& source)
{
    if (!source.remote)
        return Volume::New(source.location);
    // VC3D's cache root (the fixture wrote cache.json) holds what the run fetches.
    return Volume::NewFromUrl(source.location, {}, false);
}

namespace {

// Online reads of `location` that store what they fetch in `cacheRoot`.
vcr::zarr::OpenOptions onlineOptions(const std::filesystem::path& cacheRoot, const std::string& location)
{
    vcr::zarr::OpenOptions options;
    options.cacheDirs = {cacheRoot.string()};
    options.cacheWrites = true;
    options.access = vcr::zarr::resolveAccess(location);
    return options;
}

} // namespace

void cacheMetadata(const VolumeSource& source)
{
    // What Volume::NewFromUrl and OpenedVolumes read offline: the pyramid and each level.
    const auto options = onlineOptions(source.cacheRoot, source.location);
    for (const auto& level : vcr::zarr::pyramidLevels(source.location, options))
        (void)vcr::zarr::openPyramidLevel(source.location, level.level, options);
}

std::map<std::string, std::string> cacheSummary(CState& state, Volume&)
{
    const auto opened = state.openedVolumes();
    const auto cache = opened->cacheStats();
    const auto transfer = opened->transferStats();
    return {{"decoded_cache_mb", std::to_string(cache.bytes >> 20)},
            {"fetched_chunks", std::to_string(transfer.chunks)},
            {"fetched_mb", std::to_string(transfer.bytes >> 20)},
            {"offline_misses", std::to_string(transfer.offlineMisses)}};
}

std::uint64_t fetchedBytes(CState& state)
{
    return state.openedVolumes()->transferStats().bytes;
}

std::uint64_t offlineMisses(CState& state)
{
    return state.openedVolumes()->transferStats().offlineMisses;
}

std::pair<std::uint64_t, std::uint64_t> fetchOfflineMisses(CState& state, const std::filesystem::path& cacheRoot)
{
    std::uint64_t chunks = 0, bytes = 0;
    for (const auto& missed : state.openedVolumes()->offlineMisses()) {
        auto level = vcr::zarr::openPyramidLevel(missed.location, missed.level, onlineOptions(cacheRoot, missed.location));
        std::size_t lastReported = 0;
        const auto result = vcr::cache::prefill(*level.array, missed.chunks, nullptr,
                                                [&](std::size_t done, std::size_t total) {
            if (done == total || done - lastReported >= std::max<std::size_t>(256, total / 10)) {
                lastReported = done;
                std::fprintf(stderr, "  level %d: %zu/%zu chunks\n", missed.level, done, total);
            }
        });
        if (result.status != vcr::cache::PrefillResult::Status::Completed)
            throw std::runtime_error("fetching level " + std::to_string(missed.level) + " of " + missed.location +
                                     ": " + result.message);
        chunks += result.resolvedChunks;
        bytes += level.array->stats().remote_bytes.load();
    }
    return {chunks, bytes};
}

} // namespace vcbench::pipeline
