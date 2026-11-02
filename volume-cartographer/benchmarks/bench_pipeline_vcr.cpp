// The vc_reader render pipeline (OpenedVolumes + vcr::grid), CPU or CUDA.

#include "bench_common.hpp"

#include "CState.hpp"
#include "OpenedVolumes.hpp"
#include "VCSettings.hpp"

#include "vc/core/types/Volume.hpp"

#include <QSettings>

#include <vc_reader/cuda_available.hpp>

#include <fstream>
#include <stdexcept>

namespace vcbench::pipeline {

namespace {

std::string backend(const Args& args)
{
    const auto value = args.str("backend", "cpu");
    if (value != "cpu" && value != "cuda")
        throw std::invalid_argument("--backend must be cpu or cuda");
    return value;
}

// <root>/s3/<bucket>/<key> (vc_reader's cache layout) -> the cache root and s3://<bucket>/<key>.
std::pair<std::string, std::string> remoteSourceOf(const std::string& localPath)
{
    const auto path = std::filesystem::weakly_canonical(localPath).generic_string();
    const auto at = path.find("/s3/");
    if (at == std::string::npos)
        throw std::invalid_argument("--prefetch needs a --volume inside a vc_reader cache (<root>/s3/<bucket>/...): " +
                                    path);
    return {path.substr(0, at), "s3://" + path.substr(at + 4)};
}

} // namespace

std::string name(const Args& args)
{
    return "vc_reader-" + backend(args);
}

void writeSettings(const Args& args, const std::string& iniPath)
{
    using namespace vc3d::settings;
    QSettings settings(QString::fromStdString(iniPath), QSettings::IniFormat);
    settings.setValue(perf::RENDER_CUDA, backend(args) == "cuda");
    settings.setValue(perf::RENDER_GPU_CACHE_GB, int(args.num("gpu-cache-gb", 8)));
    settings.sync();
}

void initProcess(const Args& args)
{
    if (backend(args) == "cuda" && !vcr::cuda::available())
        throw std::runtime_error("--backend cuda: no CUDA device or a build without CUDA");
}

std::unique_ptr<CState> makeState(const Args& args)
{
    return std::make_unique<CState>(nullptr, nullptr, std::size_t(args.num("cache-gb", 16)) << 30);
}

std::shared_ptr<Volume> openVolume(const Args& args, const std::string& localPath, const std::string& configDir)
{
    if (!args.flag("prefetch"))
        return Volume::New(localPath);
    const auto [root, url] = remoteSourceOf(localPath);
    // VC3D's cache root is the cache the local folder lives in, so fetched chunks
    // land in that folder (vcr::zarr::decodedCachePath of the URL).
    std::ofstream(std::filesystem::path(configDir) / "cache.json")
        << "{\"version\": 1, \"root\": \"" << root << "\"}\n";
    return Volume::NewFromUrl(url, {}, false);
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
