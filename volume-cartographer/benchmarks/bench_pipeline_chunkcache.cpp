// VC's own render pipeline before vc_reader (origin/main): Volume's shared
// vc::render::ChunkCache, sampled by ChunkedPlaneSampler. CPU only.

#include "bench_common.hpp"

#include "CState.hpp"

#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/types/Volume.hpp"

#include <stdexcept>

namespace vcbench::pipeline {

std::string name(const Args& args)
{
    if (args.str("backend", "cpu") != "cpu")
        throw std::invalid_argument("this tree's pipeline (ChunkCache) has no --backend but cpu");
    return "chunkcache";
}

void writeSettings(const Args&, const std::string&) {}

void initProcess(const Args& args)
{
    // As VCAppMain: one process-wide service, sized by the RAM cache.
    vc::render::ChunkCacheService::Options options;
    options.decodedByteCapacity = std::size_t(args.num("cache-gb", 16)) << 30;
    vc::render::configureProcessChunkCacheService(std::move(options));
}

std::unique_ptr<CState> makeState(const Args&)
{
    return std::make_unique<CState>(nullptr);
}

std::shared_ptr<Volume> openVolume(const Args& args, const std::string& localPath, const std::string&)
{
    if (args.flag("prefetch"))
        throw std::invalid_argument("--prefetch runs on a vc_reader build; this one reads the local folder only");
    // A local volume has no persistent cache: the folder is only read.
    return Volume::New(localPath);
}

std::map<std::string, std::string> cacheSummary(CState&, Volume& volume)
{
    const auto stats = volume.sharedChunkCache()->stats();
    return {{"decoded_cache_mb", std::to_string(stats.decodedBytes >> 20)},
            {"decoded_chunks", std::to_string(stats.decodedChunks)},
            {"known_empty_chunks", std::to_string(stats.knownEmptyChunks)}};
}

std::uint64_t fetchedBytes(CState&)
{
    return 0;
}

} // namespace vcbench::pipeline
