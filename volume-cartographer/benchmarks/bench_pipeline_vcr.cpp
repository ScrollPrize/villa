// The vc_reader render pipeline (OpenedVolumes + vcr::grid), CPU or CUDA.

#include "bench_session.hpp"

#include "CState.hpp"
#include "OpenedVolumes.hpp"
#include "VCSettings.hpp"

#include "vc/core/types/Volume.hpp"

#include <QSettings>

#include <vc_reader/cuda_available.hpp>

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

std::map<std::string, std::string> cacheSummary(CState& state, Volume&)
{
    const auto opened = state.openedVolumes();
    const auto cache = opened->cacheStats();
    const auto transfer = opened->transferStats();
    return {{"decoded_cache_mb", std::to_string(cache.bytes >> 20)},
            {"fetched_chunks", std::to_string(transfer.chunks)},
            {"fetched_mb", std::to_string(transfer.bytes >> 20)}};
}

std::uint64_t fetchedBytes(CState& state)
{
    return state.openedVolumes()->transferStats().bytes;
}

} // namespace vcbench::pipeline
