// VC's own render pipeline before vc_reader (origin/main): Volume's shared
// vc::render::ChunkCache, sampled by ChunkedPlaneSampler. CPU only.

#include "bench_session.hpp"

#include "CState.hpp"

#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/types/Volume.hpp"

#include <stdexcept>

namespace vcbench::pipeline {

void writeSettings(Fixture&, const std::string&) {}

void initProcess(Fixture& fixture)
{
    // As VCAppMain: one process-wide service, sized by the RAM cache.
    vc::render::ChunkCacheService::Options options;
    options.decodedByteCapacity = std::size_t(fixture.num("cache_gb", 16)) << 30;
    vc::render::configureProcessChunkCacheService(std::move(options));
}

std::unique_ptr<CState> makeState(Fixture&)
{
    return std::make_unique<CState>(nullptr);
}

std::shared_ptr<Volume> openVolume(const VolumeSource& source)
{
    // The cached copy as a local volume: this tree has no persistent cache for it, the
    // folder is only read. (The fixture allows only cache state "cached" here.)
    if (source.remote)
        throw std::invalid_argument("this tree opens local volumes only");
    return Volume::New(source.location);
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
