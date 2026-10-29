// Computed arrays: an Array whose chunk source is a callback, persisted in the
// decoded disk-cache layout under a directory named by the computation's identity.
#include "vc_reader/computed.hpp"
#include "vc_reader/zarr.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace vcr::computed {
namespace fs = std::filesystem;
using zarr::Json;

namespace {
Json zarray(const Options& o)
{
    return Json{{"zarr_format", 2}, {"shape", o.shape}, {"chunks", o.chunks}, {"dtype", o.dtype},
                {"fill_value", o.fillValue}, {"compressor", nullptr}, {"filters", nullptr}, {"order", "C"},
                {"dimension_separator", "."}};
}

std::optional<Json> readJson(const fs::path& path)
{
    std::ifstream in(path);
    if (!in)
        return std::nullopt;
    std::stringstream text;
    text << in.rdbuf();
    return Json::parse(text.str());
}
}  // namespace

fs::path cachePath(const Options& options)
{
    if (!options.cacheDir)
        throw std::invalid_argument("computed array has no cache directory");
    if (options.identity.empty())
        throw std::invalid_argument("a persisted computed array needs an identity");
    std::uint64_t h = 1469598103934665603ull;  // FNV-1a 64
    for (const unsigned char c : options.identity) {
        h ^= c;
        h *= 1099511628211ull;
    }
    char hex[17];
    std::snprintf(hex, sizeof hex, "%016llx", static_cast<unsigned long long>(h));
    return *options.cacheDir / ("computed-" + std::string(hex));
}

std::unique_ptr<Array> open(const Options& options, ChunkCompute compute)
{
    if (!compute)
        throw std::invalid_argument("a computed array needs a compute function");
    const auto meta = zarr::parseArrayMeta(zarray(options));
    ArraySpec spec;
    spec.shape = meta.shape;
    spec.chunks = meta.chunks;
    spec.itemsize = meta.itemsize();
    spec.fill = meta.fill;
    spec.identity = "computed:" + options.identity;
    spec.threads = options.threads ? options.threads
                                   : std::min<std::size_t>(16, std::max(4u, std::thread::hardware_concurrency()));
    spec.compute = std::move(compute);
    if (options.cacheDir) {
        const fs::path dir = cachePath(options);
        const Json stored = options.compressCache ? meta.delta3dZarray() : meta.decodedZarray();
        const Json attributes = {{"vc_reader_computed", {{"identity", options.identity}}}};
        fs::create_directories(dir);
        if (auto existing = readJson(dir / ".zarray")) {
            if (*existing != stored)
                throw std::runtime_error(dir.string() + " holds computed chunks of another shape, dtype or fill: " +
                                         existing->dump());
            const auto attrs = readJson(dir / ".zattrs");
            if (!attrs || *attrs != attributes)
                throw std::runtime_error(dir.string() + " holds computed chunks of another identity");
        } else {
            zarr::atomicWrite(dir / ".zattrs", attributes.dump());
            zarr::atomicWrite(dir / ".zarray", stored.dump());
        }
        spec.layers.push_back(LayerSpec{"decoded", dir.string(), options.compressCache ? "vc-delta3d" : "", "", false});
        spec.write_layer = 0;
        if (options.cacheMaxBytes)
            spec.disk_cache = zarr::diskBudget(options.cacheDir->string(), options.cacheMaxBytes);
    }
    return std::make_unique<Array>(std::move(spec), nullptr);
}

}  // namespace vcr::computed
