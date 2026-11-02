#include "vc/core/util/CudaSampling.hpp"

#include "vc/core/util/Logging.hpp"

#include <vc_reader/array.hpp>
#include <vc_reader/cuda_available.hpp>
#include <vc_reader/zarr.hpp>
#ifdef VC_READER_CUDA
#include <vc_reader/cuda_reader.hpp>
#endif

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace vc::cuda {

namespace {

std::atomic<std::size_t> toolCacheBytes{0};
std::mutex toolReadersMutex;
struct ToolReader {
    // The reader's own handle on the level (same chunks, same memory cache): it stays
    // valid however long the caller keeps the array it opened.
    std::unique_ptr<vcr::Array> array;
    std::shared_ptr<vcr::reader_cuda::Reader> reader;
};
// By Array identity. Never destroyed: device memory must not be freed after the CUDA
// runtime's teardown.
std::map<std::string, ToolReader>& toolReaders()
{
    static auto* readers = new std::map<std::string, ToolReader>;
    return *readers;
}

#ifdef VC_READER_CUDA
// Runs a point request on the reader's device and downloads its `bytes` of results.
// False, with the reader's bricks given back, when the GPU failed.
template <class Request>
bool onDevice(vcr::reader_cuda::Reader& reader, void* out, std::size_t bytes, Request&& request)
{
    try {
        vcr::device::Guard guard(reader.device_id());
        const auto result = request();
        if (bytes) {
            vcr::device::check(cudaMemcpyAsync(out, result.storage->ptr, bytes, cudaMemcpyDeviceToHost,
                                               vcr::device::stream()));
            vcr::device::complete();
        }
        return true;
    } catch (const std::exception& error) {
        static std::atomic_bool logged{false};
        if (!logged.exchange(true))
            Logger()->warn("GPU sampling failed, sampling on the CPU where it fails again: {}", error.what());
        try {
            reader.clear();
        } catch (const std::exception&) {
        }
        return false;
    }
}
#endif

}  // namespace

std::shared_ptr<vcr::reader_cuda::Reader> makeReader(vcr::Array& array, const vcr::ElementType& type,
                                                     std::size_t cacheBytes)
{
#ifdef VC_READER_CUDA
    if (cacheBytes && vcr::cuda::available() && array.ndim() == 3 && type.kind == 'u' && type.itemsize == 1)
        return std::make_shared<vcr::reader_cuda::Reader>(array, cacheBytes, 0);
#else
    (void)array, (void)type, (void)cacheBytes;
#endif
    return nullptr;
}

void sampleLinear(vcr::Array& array, const vcr::ElementType& type, vcr::reader_cuda::Reader* reader,
                  const double* zyx, std::size_t n, float* out)
{
#ifdef VC_READER_CUDA
    if (reader && onDevice(*reader, out, n * sizeof(float), [&] {
            return reader->points(zyx, n, 64, {0, 0, 0}, 2, 1, "auto");
        }))
        return;
#else
    (void)reader;
#endif
    array.sample_linear(zyx, n, type, false, out);
}

void sampleNearest(vcr::Array& array, vcr::reader_cuda::Reader* reader, const double* zyx, std::size_t n,
                   std::uint8_t* out)
{
#ifdef VC_READER_CUDA
    if (reader && onDevice(*reader, out, n, [&] { return reader->points(zyx, n, 64, {0, 0, 0}, 2, 0, "auto"); }))
        return;
#else
    (void)reader;
#endif
    array.sample_nearest(zyx, n, out);
}

void readVoxels(vcr::reader_cuda::Reader& reader, const std::int64_t* zyx, std::size_t n, std::uint8_t* out)
{
#ifdef VC_READER_CUDA
    if (!onDevice(reader, out, n, [&] { return reader.points(zyx, n, 64, {0, 0, 0}, 3, 0, "auto"); }))
        throw std::runtime_error("GPU voxel read failed");
#else
    (void)reader, (void)zyx, (void)n, (void)out;
    throw std::logic_error("no CUDA reader in this build");
#endif
}

void enableForTool(std::size_t cacheBytes)
{
#ifndef VC_READER_CUDA
    (void)cacheBytes;
    throw std::runtime_error("--cuda: this build has no CUDA support");
#else
    if (!vcr::cuda::available())
        throw std::runtime_error("--cuda: no CUDA device is available");
    if (!cacheBytes)
        throw std::runtime_error("--cuda-cache-gb must be positive");
    toolCacheBytes = cacheBytes;
#endif
}

bool enabledForTool() { return toolCacheBytes.load() != 0; }

void applyToolOptions(int& argc, char** argv)
{
    bool enable = false;
    std::size_t gb = kDefaultToolCacheGb;
    int kept = 1;
    try {
        for (int i = 1; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--cuda")) {
                enable = true;
            } else if (!std::strcmp(argv[i], "--cuda-cache-gb")) {
                if (i + 1 == argc)
                    throw std::runtime_error("--cuda-cache-gb requires a value");
                const long long value = std::stoll(argv[++i]);
                if (value <= 0)
                    throw std::runtime_error("--cuda-cache-gb must be positive");
                gb = std::size_t(value);
                enable = true;
            } else {
                argv[kept++] = argv[i];
            }
        }
        argc = kept;
        argv[argc] = nullptr;
        if (enable)
            enableForTool(gb << 30);
    } catch (const std::exception& error) {
        std::cerr << argv[0] << ": " << error.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

const char* toolOptionsHelp()
{
    return "  --cuda                          sample uint8 3-D volumes on the GPU (fails without a device)\n"
           "  --cuda-cache-gb N               device memory per opened level, implies --cuda [4]\n";
}

vcr::reader_cuda::Reader* toolReader(const vcr::zarr::OpenedArray& level)
{
    const auto bytes = toolCacheBytes.load();
    if (!bytes)
        return nullptr;
    std::lock_guard lock(toolReadersMutex);
    const auto& spec = level.array->spec();
    auto [entry, fresh] = toolReaders().try_emplace(spec.identity);
    if (fresh) {
        entry->second.array = std::make_unique<vcr::Array>(spec, level.array->memory_cache());
        entry->second.reader = makeReader(*entry->second.array, level.meta.type, bytes);
        if (!entry->second.reader)
            Logger()->warn("--cuda: {} stays on the CPU: the GPU samples 3-D uint8 volumes, this is {}-D {}",
                           level.location, level.meta.shape.size(), level.meta.dtype);
    }
    return entry->second.reader.get();
}

vcr::reader_cuda::Reader* toolReaderOf(const vcr::Array& array)
{
    if (!toolCacheBytes.load())
        return nullptr;
    std::lock_guard lock(toolReadersMutex);
    const auto found = toolReaders().find(array.spec().identity);
    return found == toolReaders().end() ? nullptr : found->second.reader.get();
}

}  // namespace vc::cuda
