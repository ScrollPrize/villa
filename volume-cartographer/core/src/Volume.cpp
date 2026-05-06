#include "vc/core/types/Volume.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#include <opencv2/imgcodecs.hpp>
#include "vc/core/util/LoadJson.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/render/ZarrChunkFetcher.hpp"
#include "vc/core/util/HttpFetch.hpp"
#include "vc/core/util/RemoteUrl.hpp"
#include "vc/core/util/PostProcess.hpp"
#include "vc/core/render/IChunkedArray.hpp"
#include "vc/core/render/ChunkFetch.hpp"
#include "utils/hash.hpp"
#include "utils/zarr.hpp"

static const std::filesystem::path METADATA_FILE = "meta.json";
static const std::filesystem::path METADATA_FILE_ALT = "metadata.json";

namespace
{

bool isRemoteAuthError(const std::exception& e)
{
    const std::string msg = e.what();
    return msg.find("AWS credentials") != std::string::npos ||
           msg.find("Access denied") != std::string::npos ||
           msg.find("ExpiredToken") != std::string::npos ||
           msg.find("InvalidToken") != std::string::npos ||
           msg.find("TokenRefreshRequired") != std::string::npos ||
           msg.find("InvalidAccessKeyId") != std::string::npos ||
           msg.find("SignatureDoesNotMatch") != std::string::npos ||
           msg.find("HTTP 400") != std::string::npos ||
           msg.find("HTTP 401") != std::string::npos ||
           msg.find("HTTP 403") != std::string::npos;
}

std::string normalizeRemoteVolumeUrl(std::string url)
{
    while (!url.empty() && url.back() == '/')
        url.pop_back();
    return url;
}

std::string deriveRemoteVolumeName(const std::string& url)
{
    const auto normalized = normalizeRemoteVolumeUrl(url);
    const auto pos = normalized.rfind('/');
    if (pos != std::string::npos && pos + 1 < normalized.size())
        return normalized.substr(pos + 1);
    return normalized.empty() ? std::string("remote") : normalized;
}

std::optional<utils::Json> loadRemoteVolumeMetadata(const std::string& remoteUrl,
                                                    const vc::HttpAuth& auth)
{
    const auto load = [&](const std::string& name) -> std::optional<utils::Json> {
        const auto url = remoteUrl + "/" + name;
        const auto body = vc::httpGetString(url, auth);
        if (body.empty()) {
            return std::nullopt;
        }
        auto json = utils::Json::parse(body);
        if (name == METADATA_FILE_ALT.string()) {
            if (!json.contains("scan")) {
                throw std::runtime_error("metadata.json missing 'scan' key: " + url);
            }
            json.update(json["scan"]);
            if (!json.contains("format")) {
                json["format"] = "zarr";
            }
        }
        if (!json.is_object()) {
            throw std::runtime_error("remote volume metadata is not an object: " + url);
        }
        return json;
    };

    if (auto meta = load(METADATA_FILE.string())) {
        return meta;
    }
    return load(METADATA_FILE_ALT.string());
}

std::string deriveRemoteVolumeId(const std::string& url)
{
    const auto normalized = normalizeRemoteVolumeUrl(url);
    const auto name = deriveRemoteVolumeName(normalized);
    const auto hash = utils::fnv1a(std::string_view(normalized));

    std::ostringstream out;
    out << name << "-" << std::hex << std::nouppercase << std::setw(16)
        << std::setfill('0') << hash;
    return out.str();
}

std::vector<vc::render::ChunkCache::LevelInfo>
makeChunkCacheLevelInfo(const vc::render::OpenedChunkedZarr& opened)
{
    std::vector<vc::render::ChunkCache::LevelInfo> levels;
    levels.reserve(opened.fetchers.size());
    for (std::size_t i = 0; i < opened.fetchers.size(); ++i) {
        vc::render::ChunkCache::LevelInfo level;
        level.shape = opened.shapes[i];
        level.chunkShape = opened.chunkShapes[i];
        level.transform = opened.transforms[i];
        levels.push_back(level);
    }
    return levels;
}

} // namespace

namespace {

template <typename T>
T typedFillValue(vc::render::IChunkedArray& array)
{
    const double fill = array.fillValue();
    if constexpr (std::is_same_v<T, uint8_t>) {
        return static_cast<uint8_t>(std::clamp(fill, 0.0, 255.0));
    } else {
        return static_cast<uint16_t>(std::clamp(fill, 0.0, 65535.0));
    }
}

template <typename T>
void fillFromChunkedArrayFillValue(Array3D<T>& out, vc::render::IChunkedArray& array)
{
    out.fill(typedFillValue<T>(array));
}

template <typename T>
void readFromChunkedArrayZYX(Array3D<T>& out,
                                   const std::array<int, 3>& offsetZYX,
                                   vc::render::IChunkedArray& array,
                                   int level)
{
    using vc::render::ChunkKey;
    using vc::render::ChunkStatus;
    using vc::render::ChunkDtype;

    const ChunkDtype expectedDtype = std::is_same_v<T, uint8_t>
        ? ChunkDtype::UInt8
        : ChunkDtype::UInt16;
    if (array.dtype() != expectedDtype) {
        throw std::runtime_error("Volume::read dtype does not match volume dtype");
    }

    const auto outShape = out.shape();
    if (outShape[0] == 0 || outShape[1] == 0 || outShape[2] == 0) {
        return;
    }
    out.fill(T{});

    const auto volumeShape = array.shape(level);
    const auto chunkShape = array.chunkShape(level);
    if (chunkShape[0] <= 0 || chunkShape[1] <= 0 || chunkShape[2] <= 0) {
        throw std::runtime_error("Volume::read encountered invalid chunk shape");
    }

    const int z0 = offsetZYX[0];
    const int y0 = offsetZYX[1];
    const int x0 = offsetZYX[2];
    const int z1 = z0 + static_cast<int>(outShape[0]) - 1;
    const int y1 = y0 + static_cast<int>(outShape[1]) - 1;
    const int x1 = x0 + static_cast<int>(outShape[2]) - 1;

    const int readZ0 = std::max(0, z0);
    const int readY0 = std::max(0, y0);
    const int readX0 = std::max(0, x0);
    const int readZ1 = std::min(volumeShape[0] - 1, z1);
    const int readY1 = std::min(volumeShape[1] - 1, y1);
    const int readX1 = std::min(volumeShape[2] - 1, x1);

    if (readZ0 <= readZ1 && readY0 <= readY1 && readX0 <= readX1) {
        std::vector<ChunkKey> keys;
        const int cZ0 = readZ0 / chunkShape[0];
        const int cY0 = readY0 / chunkShape[1];
        const int cX0 = readX0 / chunkShape[2];
        const int cZ1 = readZ1 / chunkShape[0];
        const int cY1 = readY1 / chunkShape[1];
        const int cX1 = readX1 / chunkShape[2];
        keys.reserve(static_cast<size_t>(cZ1 - cZ0 + 1) *
                     static_cast<size_t>(cY1 - cY0 + 1) *
                     static_cast<size_t>(cX1 - cX0 + 1));
        for (int cz = cZ0; cz <= cZ1; ++cz) {
            for (int cy = cY0; cy <= cY1; ++cy) {
                for (int cx = cX0; cx <= cX1; ++cx) {
                    keys.push_back({level, cz, cy, cx});
                }
            }
        }
        if (!keys.empty()) {
            array.prefetchChunks(keys, false);
        }
    }

    if (readZ0 > readZ1 || readY0 > readY1 || readX0 > readX1) {
        return;
    }

    struct ChunkCopyPlan {
        int cz = 0;
        int cy = 0;
        int cx = 0;
        int zBegin = 0;
        int zEnd = 0;
        int yBegin = 0;
        int yEnd = 0;
        int xBegin = 0;
        int xEnd = 0;
    };

    const int cZ0 = readZ0 / chunkShape[0];
    const int cY0 = readY0 / chunkShape[1];
    const int cX0 = readX0 / chunkShape[2];
    const int cZ1 = readZ1 / chunkShape[0];
    const int cY1 = readY1 / chunkShape[1];
    const int cX1 = readX1 / chunkShape[2];

    std::vector<ChunkCopyPlan> plans;
    plans.reserve(static_cast<size_t>(cZ1 - cZ0 + 1) *
                  static_cast<size_t>(cY1 - cY0 + 1) *
                  static_cast<size_t>(cX1 - cX0 + 1));
    for (int cz = cZ0; cz <= cZ1; ++cz) {
        const int chunkZ0 = cz * chunkShape[0];
        const int chunkZ1 = std::min(volumeShape[0], chunkZ0 + chunkShape[0]);
        for (int cy = cY0; cy <= cY1; ++cy) {
            const int chunkY0 = cy * chunkShape[1];
            const int chunkY1 = std::min(volumeShape[1], chunkY0 + chunkShape[1]);
            for (int cx = cX0; cx <= cX1; ++cx) {
                const int chunkX0 = cx * chunkShape[2];
                const int chunkX1 = std::min(volumeShape[2], chunkX0 + chunkShape[2]);
                plans.push_back(ChunkCopyPlan{
                    cz,
                    cy,
                    cx,
                    std::max(readZ0, chunkZ0),
                    std::min(readZ1 + 1, chunkZ1),
                    std::max(readY0, chunkY0),
                    std::min(readY1 + 1, chunkY1),
                    std::max(readX0, chunkX0),
                    std::min(readX1 + 1, chunkX1),
                });
            }
        }
    }

    const T fill = typedFillValue<T>(array);
    const size_t chunkStrideY = static_cast<size_t>(chunkShape[2]);
    const size_t chunkStrideZ = static_cast<size_t>(chunkShape[1]) * chunkStrideY;
    const size_t outStrideY = outShape[0];
    const size_t outStrideX = outShape[0] * outShape[1];
    T* outData = out.data();
    std::exception_ptr firstException;
    std::mutex exceptionMutex;

    #pragma omp parallel for schedule(dynamic)
    for (size_t planIndex = 0; planIndex < plans.size(); ++planIndex) {
        try {
            const auto& plan = plans[planIndex];
            const auto result = array.getChunkBlocking(level, plan.cz, plan.cy, plan.cx);
            if (result.status == ChunkStatus::Error) {
                throw std::runtime_error(result.error.empty() ? "chunk fetch failed" : result.error);
            }

            const T* chunkData = nullptr;
            std::shared_ptr<const std::vector<std::byte>> chunkBytes;
            if (result.status == ChunkStatus::Data && result.bytes) {
                chunkBytes = result.bytes;
                chunkData = reinterpret_cast<const T*>(chunkBytes->data());
            }

            const int chunkZ0 = plan.cz * chunkShape[0];
            const int chunkY0 = plan.cy * chunkShape[1];
            const int chunkX0 = plan.cx * chunkShape[2];

            for (int x = plan.xBegin; x < plan.xEnd; ++x) {
                const size_t outX = static_cast<size_t>(x - x0);
                const int lx = x - chunkX0;
                for (int y = plan.yBegin; y < plan.yEnd; ++y) {
                    const size_t outY = static_cast<size_t>(y - y0);
                    const int ly = y - chunkY0;
                    T* dst = outData + outX * outStrideX + outY * outStrideY +
                             static_cast<size_t>(plan.zBegin - z0);

                    if (chunkData == nullptr ||
                        result.status == ChunkStatus::AllFill ||
                        result.status == ChunkStatus::Missing) {
                        std::fill(dst, dst + (plan.zEnd - plan.zBegin), fill);
                        continue;
                    }

                    const int lzBegin = plan.zBegin - chunkZ0;
                    for (int z = plan.zBegin; z < plan.zEnd; ++z) {
                        const int lz = lzBegin + (z - plan.zBegin);
                        const size_t srcOffset =
                            static_cast<size_t>(lz) * chunkStrideZ +
                            static_cast<size_t>(ly) * chunkStrideY +
                            static_cast<size_t>(lx);
                        dst[z - plan.zBegin] = chunkData[srcOffset];
                    }
                }
            }
        } catch (...) {
            std::lock_guard lock(exceptionMutex);
            if (!firstException) {
                firstException = std::current_exception();
            }
        }
    }

    if (firstException) {
        std::rethrow_exception(firstException);
    }
}

template <typename T>
void downsampleMeanZYX(Array3D<T>& out,
                       const Array3D<T>& src,
                       const std::array<int, 3>& srcOffsetZYX,
                       const std::array<int, 3>& srcVolumeShapeZYX,
                       int factor)
{
    const auto outShape = out.shape();
    const auto srcShape = src.shape();
    const std::uint64_t denom = static_cast<std::uint64_t>(factor) *
                                static_cast<std::uint64_t>(factor) *
                                static_cast<std::uint64_t>(factor);

    #pragma omp parallel for collapse(2) schedule(static)
    for (std::int64_t z = 0; z < static_cast<std::int64_t>(outShape[0]); ++z) {
        for (std::int64_t y = 0; y < static_cast<std::int64_t>(outShape[1]); ++y) {
            for (std::size_t x = 0; x < outShape[2]; ++x) {
                std::uint64_t sum = 0;
                const std::size_t srcZ0 = static_cast<std::size_t>(z) * static_cast<std::size_t>(factor);
                const std::size_t srcY0 = static_cast<std::size_t>(y) * static_cast<std::size_t>(factor);
                const std::size_t srcX0 = x * static_cast<std::size_t>(factor);
                for (int dz = 0; dz < factor; ++dz) {
                    for (int dy = 0; dy < factor; ++dy) {
                        for (int dx = 0; dx < factor; ++dx) {
                            const int absZ = std::clamp(
                                srcOffsetZYX[0] + static_cast<int>(srcZ0) + dz,
                                0,
                                srcVolumeShapeZYX[0] - 1);
                            const int absY = std::clamp(
                                srcOffsetZYX[1] + static_cast<int>(srcY0) + dy,
                                0,
                                srcVolumeShapeZYX[1] - 1);
                            const int absX = std::clamp(
                                srcOffsetZYX[2] + static_cast<int>(srcX0) + dx,
                                0,
                                srcVolumeShapeZYX[2] - 1);
                            const std::size_t sz = static_cast<std::size_t>(absZ - srcOffsetZYX[0]);
                            const std::size_t sy = static_cast<std::size_t>(absY - srcOffsetZYX[1]);
                            const std::size_t sx = static_cast<std::size_t>(absX - srcOffsetZYX[2]);
                            if (sz < srcShape[0] && sy < srcShape[1] && sx < srcShape[2]) {
                                sum += src(sz, sy, sx);
                            }
                        }
                    }
                }
                out(static_cast<std::size_t>(z), static_cast<std::size_t>(y), x) =
                    static_cast<T>((sum + denom / 2) / denom);
            }
        }
    }
}

std::array<int, 3> xyzToZyx(const std::array<int, 3>& xyz)
{
    return {xyz[2], xyz[1], xyz[0]};
}

utils::ZarrDtype zarrDtypeFromChunkDtype(vc::render::ChunkDtype dtype)
{
    switch (dtype) {
    case vc::render::ChunkDtype::UInt8:
        return utils::ZarrDtype::uint8;
    case vc::render::ChunkDtype::UInt16:
        return utils::ZarrDtype::uint16;
    }
    throw std::runtime_error("unsupported Volume zarr dtype");
}

vc::render::ChunkDtype chunkDtypeFromZarr(utils::ZarrDtype dtype)
{
    if (dtype == utils::ZarrDtype::uint8)
        return vc::render::ChunkDtype::UInt8;
    if (dtype == utils::ZarrDtype::uint16)
        return vc::render::ChunkDtype::UInt16;
    throw std::runtime_error("Volume zarr writes only support uint8 and uint16");
}

template <typename T>
vc::render::ChunkDtype chunkDtypeFor()
{
    if constexpr (std::is_same_v<T, uint8_t>)
        return vc::render::ChunkDtype::UInt8;
    else
        return vc::render::ChunkDtype::UInt16;
}

template <typename T>
T typedFillValue(double fill)
{
    if constexpr (std::is_same_v<T, uint8_t>)
        return static_cast<uint8_t>(std::clamp(fill, 0.0, 255.0));
    else
        return static_cast<uint16_t>(std::clamp(fill, 0.0, 65535.0));
}

utils::ZarrArray::Codec zarrCodecFor(const std::string& compressor, int level)
{
    if (compressor.empty() || compressor == "none")
        return {};
#if UTILS_HAS_COMPRESSION
    return utils::make_zarr_codec(compressor, level);
#else
    throw std::runtime_error("zarr compression support is unavailable for compressor: " + compressor);
#endif
}

utils::ZarrArray::CodecRegistry zarrCodecRegistry()
{
    utils::ZarrArray::CodecRegistry registry;
#if UTILS_HAS_COMPRESSION
    for (const char* name : {"blosc", "zstd", "gzip", "zlib", "lz4"}) {
        try {
            registry.emplace(name, utils::make_zarr_codec(name));
        } catch (...) {
        }
    }
#endif
    return registry;
}

utils::ZarrArray openLocalZarrArrayForWrite(const std::filesystem::path& path)
{
    auto registry = zarrCodecRegistry();
    if (!registry.empty())
        return utils::ZarrArray::open(path, std::move(registry));

    auto array = utils::ZarrArray::open(path);
    const auto& meta = array.metadata();
    if (!meta.compressor_id.empty() || !meta.codecs.empty())
        throw std::runtime_error("cannot write compressed zarr without compression codec support: " + path.string());
    return array;
}

std::vector<size_t> toVector(const std::array<size_t, 3>& value)
{
    return {value[0], value[1], value[2]};
}

void validatePyramidPolicy(const Volume::PyramidPolicy& policy)
{
    for (double scale : policy.downsampleZYX) {
        if (!std::isfinite(scale) || scale < 1.0) {
            throw std::runtime_error("Volume pyramid downsample values must be finite and >= 1.0");
        }
    }
}

size_t downsampledDim(size_t dim, double scale)
{
    if (scale <= 1.0)
        return dim;
    return std::max<size_t>(1, static_cast<size_t>(std::ceil(static_cast<double>(dim) / scale)));
}

std::array<size_t, 3> downsampledShape(const std::array<size_t, 3>& shape,
                                       const Volume::PyramidPolicy& policy)
{
    return {
        downsampledDim(shape[0], policy.downsampleZYX[0]),
        downsampledDim(shape[1], policy.downsampleZYX[1]),
        downsampledDim(shape[2], policy.downsampleZYX[2]),
    };
}

std::array<size_t, 3> clampChunkShape(const std::array<size_t, 3>& chunkShape,
                                      const std::array<size_t, 3>& shape)
{
    return {
        std::max<size_t>(1, std::min(chunkShape[0], shape[0])),
        std::max<size_t>(1, std::min(chunkShape[1], shape[1])),
        std::max<size_t>(1, std::min(chunkShape[2], shape[2])),
    };
}

template <typename T>
void fillRawChunk(std::vector<std::byte>& chunkBytes, T fill)
{
    auto* typed = reinterpret_cast<T*>(chunkBytes.data());
    std::fill(typed, typed + chunkBytes.size() / sizeof(T), fill);
}

template <typename T>
void readZarrRegionZYX(utils::ZarrArray& array,
                       Array3D<T>& out,
                       const std::array<size_t, 3>& offsetZYX)
{
    const auto& meta = array.metadata();
    if (chunkDtypeFromZarr(meta.dtype) != chunkDtypeFor<T>())
        throw std::runtime_error("Volume::read/write dtype does not match zarr dtype");
    if (meta.shape.size() != 3 || meta.chunks.size() != 3)
        throw std::runtime_error("Volume zarr region I/O requires 3D arrays");

    const auto outShape = out.shape();
    if (outShape[0] == 0 || outShape[1] == 0 || outShape[2] == 0)
        return;

    const T fill = typedFillValue<T>(meta.fill_value.value_or(0.0));
    out.fill(fill);

    std::array<size_t, 3> shape{meta.shape[0], meta.shape[1], meta.shape[2]};
    std::array<size_t, 3> chunks{meta.chunks[0], meta.chunks[1], meta.chunks[2]};
    for (size_t d = 0; d < 3; ++d) {
        if (offsetZYX[d] >= shape[d])
            return;
    }

    const std::array<size_t, 3> endZYX{
        std::min(shape[0], offsetZYX[0] + outShape[0]),
        std::min(shape[1], offsetZYX[1] + outShape[1]),
        std::min(shape[2], offsetZYX[2] + outShape[2]),
    };
    if (offsetZYX[0] >= endZYX[0] || offsetZYX[1] >= endZYX[1] || offsetZYX[2] >= endZYX[2])
        return;

    const size_t chunkBytes = chunks[0] * chunks[1] * chunks[2] * sizeof(T);
    const size_t chunkStrideY = chunks[2];
    const size_t chunkStrideZ = chunks[1] * chunkStrideY;

    for (size_t cz = offsetZYX[0] / chunks[0]; cz <= (endZYX[0] - 1) / chunks[0]; ++cz) {
        const size_t chunkBaseZ = cz * chunks[0];
        for (size_t cy = offsetZYX[1] / chunks[1]; cy <= (endZYX[1] - 1) / chunks[1]; ++cy) {
            const size_t chunkBaseY = cy * chunks[1];
            for (size_t cx = offsetZYX[2] / chunks[2]; cx <= (endZYX[2] - 1) / chunks[2]; ++cx) {
                const size_t chunkBaseX = cx * chunks[2];
                const std::array<size_t, 3> indices{cz, cy, cx};
                auto bytes = array.read_chunk(indices);
                const T* src = nullptr;
                if (bytes && bytes->size() >= chunkBytes)
                    src = reinterpret_cast<const T*>(bytes->data());

                const size_t z0 = std::max(chunkBaseZ, offsetZYX[0]);
                const size_t y0 = std::max(chunkBaseY, offsetZYX[1]);
                const size_t x0 = std::max(chunkBaseX, offsetZYX[2]);
                const size_t z1 = std::min(chunkBaseZ + chunks[0], endZYX[0]);
                const size_t y1 = std::min(chunkBaseY + chunks[1], endZYX[1]);
                const size_t x1 = std::min(chunkBaseX + chunks[2], endZYX[2]);

                for (size_t z = z0; z < z1; ++z) {
                    for (size_t y = y0; y < y1; ++y) {
                        for (size_t x = x0; x < x1; ++x) {
                            const size_t dstZ = z - offsetZYX[0];
                            const size_t dstY = y - offsetZYX[1];
                            const size_t dstX = x - offsetZYX[2];
                            if (src) {
                                const size_t srcOff =
                                    (z - chunkBaseZ) * chunkStrideZ +
                                    (y - chunkBaseY) * chunkStrideY +
                                    (x - chunkBaseX);
                                out(dstZ, dstY, dstX) = src[srcOff];
                            }
                        }
                    }
                }
            }
        }
    }
}

size_t zarrDtypeSize(const utils::ZarrArray& array)
{
    return utils::dtype_size(array.metadata().dtype);
}

std::vector<std::byte> filledChunkBytes(const utils::ZarrArray& array)
{
    const size_t chunkElems = array.metadata().sub_chunk_byte_size() / zarrDtypeSize(array);
    const size_t dtypeSize = zarrDtypeSize(array);
    std::vector<std::byte> out(chunkElems * dtypeSize);
    const double fill = array.metadata().fill_value.value_or(0.0);
    switch (array.metadata().dtype) {
    case utils::ZarrDtype::uint8:
        fillRawChunk<uint8_t>(out, typedFillValue<uint8_t>(fill));
        break;
    case utils::ZarrDtype::uint16:
        fillRawChunk<uint16_t>(out, typedFillValue<uint16_t>(fill));
        break;
    default:
        throw std::runtime_error("Volume chunk fill only supports uint8 and uint16");
    }
    return out;
}

bool chunkMatchesFill(const utils::ZarrArray& array, std::span<const std::byte> data)
{
    const auto fill = filledChunkBytes(array);
    return data.size() == fill.size() &&
           std::equal(data.begin(), data.end(), fill.begin());
}

template <typename T>
void writeZarrRegionZYX(utils::ZarrArray& array,
                        const Array3D<T>& data,
                        const std::array<size_t, 3>& offsetZYX)
{
    const auto& meta = array.metadata();
    if (chunkDtypeFromZarr(meta.dtype) != chunkDtypeFor<T>())
        throw std::runtime_error("Volume::write dtype does not match zarr dtype");
    if (meta.shape.size() != 3 || meta.chunks.size() != 3)
        throw std::runtime_error("Volume zarr writes require 3D arrays");

    const auto regionShape = data.shape();
    if (regionShape[0] == 0 || regionShape[1] == 0 || regionShape[2] == 0)
        return;

    std::array<size_t, 3> shape{meta.shape[0], meta.shape[1], meta.shape[2]};
    std::array<size_t, 3> chunks{meta.chunks[0], meta.chunks[1], meta.chunks[2]};
    for (size_t d = 0; d < 3; ++d) {
        if (offsetZYX[d] > shape[d] || regionShape[d] > shape[d] - offsetZYX[d])
            throw std::out_of_range("Volume::write region exceeds zarr bounds");
        if (chunks[d] == 0)
            throw std::runtime_error("Volume::write encountered invalid zarr chunk shape");
    }

    const T fill = typedFillValue<T>(meta.fill_value.value_or(0.0));
    const size_t chunkElems = chunks[0] * chunks[1] * chunks[2];
    const size_t chunkBytes = chunkElems * sizeof(T);
    const size_t chunkStrideY = chunks[2];
    const size_t chunkStrideZ = chunks[1] * chunkStrideY;
    std::vector<std::byte> chunkBuf(chunkBytes);

    const std::array<size_t, 3> endZYX{
        offsetZYX[0] + regionShape[0],
        offsetZYX[1] + regionShape[1],
        offsetZYX[2] + regionShape[2],
    };

    for (size_t cz = offsetZYX[0] / chunks[0]; cz <= (endZYX[0] - 1) / chunks[0]; ++cz) {
        const size_t chunkBaseZ = cz * chunks[0];
        for (size_t cy = offsetZYX[1] / chunks[1]; cy <= (endZYX[1] - 1) / chunks[1]; ++cy) {
            const size_t chunkBaseY = cy * chunks[1];
            for (size_t cx = offsetZYX[2] / chunks[2]; cx <= (endZYX[2] - 1) / chunks[2]; ++cx) {
                const size_t chunkBaseX = cx * chunks[2];
                const std::array<size_t, 3> indices{cz, cy, cx};

                const size_t z0 = std::max(chunkBaseZ, offsetZYX[0]);
                const size_t y0 = std::max(chunkBaseY, offsetZYX[1]);
                const size_t x0 = std::max(chunkBaseX, offsetZYX[2]);
                const size_t z1 = std::min(chunkBaseZ + chunks[0], endZYX[0]);
                const size_t y1 = std::min(chunkBaseY + chunks[1], endZYX[1]);
                const size_t x1 = std::min(chunkBaseX + chunks[2], endZYX[2]);

                const bool fullChunk =
                    z0 == chunkBaseZ && y0 == chunkBaseY && x0 == chunkBaseX &&
                    z1 == chunkBaseZ + chunks[0] &&
                    y1 == chunkBaseY + chunks[1] &&
                    x1 == chunkBaseX + chunks[2];

                if (!fullChunk) {
                    auto existing = array.read_chunk(indices);
                    if (existing && existing->size() >= chunkBytes) {
                        std::memcpy(chunkBuf.data(), existing->data(), chunkBytes);
                    } else {
                        fillRawChunk(chunkBuf, fill);
                    }
                } else {
                    fillRawChunk(chunkBuf, fill);
                }

                auto* dst = reinterpret_cast<T*>(chunkBuf.data());
                for (size_t z = z0; z < z1; ++z) {
                    for (size_t y = y0; y < y1; ++y) {
                        for (size_t x = x0; x < x1; ++x) {
                            const size_t srcZ = z - offsetZYX[0];
                            const size_t srcY = y - offsetZYX[1];
                            const size_t srcX = x - offsetZYX[2];
                            const size_t dstOff =
                                (z - chunkBaseZ) * chunkStrideZ +
                                (y - chunkBaseY) * chunkStrideY +
                                (x - chunkBaseX);
                            dst[dstOff] = data(srcZ, srcY, srcX);
                        }
                    }
                }

                array.write_chunk(indices, std::span<const std::byte>(chunkBuf.data(), chunkBuf.size()));
            }
        }
    }
}

template <typename T>
void downsampleZYX(Array3D<T>& out,
                   const Array3D<T>& src,
                   const std::array<size_t, 3>& srcOffsetZYX,
                   const std::array<size_t, 3>& srcVolumeShapeZYX,
                   const std::array<double, 3>& scaleZYX,
                   Volume::PyramidPolicy::Reduction reduction)
{
    const auto outShape = out.shape();
    const auto srcShape = src.shape();

    #pragma omp parallel for collapse(2) schedule(static)
    for (std::int64_t z = 0; z < static_cast<std::int64_t>(outShape[0]); ++z) {
        for (std::int64_t y = 0; y < static_cast<std::int64_t>(outShape[1]); ++y) {
            for (std::size_t x = 0; x < outShape[2]; ++x) {
                const std::array<size_t, 3> dstAbs{
                    static_cast<size_t>(z),
                    static_cast<size_t>(y),
                    x,
                };
                std::array<size_t, 3> srcBegin{};
                std::array<size_t, 3> srcEnd{};
                for (size_t d = 0; d < 3; ++d) {
                    const double begin = static_cast<double>(dstAbs[d]) * scaleZYX[d];
                    const double end = static_cast<double>(dstAbs[d] + 1) * scaleZYX[d];
                    srcBegin[d] = std::min(srcVolumeShapeZYX[d],
                                           static_cast<size_t>(std::floor(begin)));
                    srcEnd[d] = std::min(srcVolumeShapeZYX[d],
                                         static_cast<size_t>(std::ceil(end)));
                    if (srcEnd[d] <= srcBegin[d] && srcBegin[d] < srcVolumeShapeZYX[d])
                        srcEnd[d] = srcBegin[d] + 1;
                }

                std::uint64_t sum = 0;
                std::uint64_t count = 0;
                T maxValue = T{};
                for (size_t szAbs = srcBegin[0]; szAbs < srcEnd[0]; ++szAbs) {
                    const size_t sz = szAbs - srcOffsetZYX[0];
                    if (sz >= srcShape[0])
                        continue;
                    for (size_t syAbs = srcBegin[1]; syAbs < srcEnd[1]; ++syAbs) {
                        const size_t sy = syAbs - srcOffsetZYX[1];
                        if (sy >= srcShape[1])
                            continue;
                        for (size_t sxAbs = srcBegin[2]; sxAbs < srcEnd[2]; ++sxAbs) {
                            const size_t sx = sxAbs - srcOffsetZYX[2];
                            if (sx >= srcShape[2])
                                continue;
                            const T value = src(sz, sy, sx);
                            sum += value;
                            maxValue = std::max(maxValue, value);
                            ++count;
                        }
                    }
                }
                T result = T{};
                if (count != 0) {
                    switch (reduction) {
                    case Volume::PyramidPolicy::Reduction::Mean:
                        result = static_cast<T>((sum + count / 2) / count);
                        break;
                    case Volume::PyramidPolicy::Reduction::Max:
                        result = maxValue;
                        break;
                    case Volume::PyramidPolicy::Reduction::BinaryOr:
                        result = maxValue == T{} ? T{} : T{1};
                        break;
                    }
                }
                out(static_cast<size_t>(z), static_cast<size_t>(y), x) = result;
            }
        }
    }
}

void writeTextFile(const std::filesystem::path& path, const std::string& text)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("failed to open for write: " + path.string());
    out << text;
    if (!out)
        throw std::runtime_error("failed to write: " + path.string());
}

void createLocalOmeZarr(const std::filesystem::path& path,
                        const Volume::ZarrCreateOptions& options)
{
    if (options.shapeZYX[0] == 0 || options.shapeZYX[1] == 0 || options.shapeZYX[2] == 0)
        throw std::runtime_error("Volume::New zarr creation requires a non-empty shapeZYX");
    if (options.numLevels == 0)
        throw std::runtime_error("Volume::New zarr creation requires at least one level");
    validatePyramidPolicy(options.pyramid);

    if (options.overwriteExisting) {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
        if (ec)
            throw std::runtime_error("failed to remove existing zarr path: " + path.string());
    }

    std::filesystem::create_directories(path);
    writeTextFile(path / ".zgroup", R"({"zarr_format": 2})" "\n");

    const auto dtype = zarrDtypeFromChunkDtype(options.dtype);
    std::array<size_t, 3> levelShape = options.shapeZYX;
    for (size_t level = 0; level < options.numLevels; ++level) {
        utils::ZarrMetadata meta;
        meta.version = utils::ZarrVersion::v2;
        meta.shape = toVector(levelShape);
        meta.chunks = toVector(clampChunkShape(options.chunkShapeZYX, levelShape));
        meta.dtype = dtype;
        meta.fill_value = options.fillValue;
        meta.dimension_separator = ".";
        if (!options.compressor.empty() && options.compressor != "none") {
            meta.compressor_id = options.compressor;
            meta.compression_level = options.compressionLevel;
        }
        utils::ZarrArray::create(path / std::to_string(level),
                                 std::move(meta),
                                 zarrCodecFor(options.compressor, options.compressionLevel));
        levelShape = downsampledShape(levelShape, options.pyramid);
    }

    const auto baseName = path.filename().string();
    const std::string uuid = options.uuid.empty() ? baseName : options.uuid;
    const std::string name = options.name.empty() ? baseName : options.name;

    utils::Json metadata;
    metadata["type"] = "vol";
    metadata["uuid"] = uuid;
    metadata["name"] = name;
    metadata["format"] = "zarr";
    metadata["width"] = static_cast<int>(options.shapeZYX[2]);
    metadata["height"] = static_cast<int>(options.shapeZYX[1]);
    metadata["slices"] = static_cast<int>(options.shapeZYX[0]);
    metadata["voxelsize"] = options.voxelSize;
    metadata["min"] = 0.0;
    metadata["max"] = options.dtype == vc::render::ChunkDtype::UInt8 ? 255.0 : 65535.0;
    writeTextFile(path / METADATA_FILE, metadata.dump(2) + "\n");

    utils::Json attrs;
    attrs["note_axes_order"] = "ZYX (slice, row, col)";
    utils::Json multiscale;
    multiscale["version"] = "0.4";
    multiscale["name"] = name;
    utils::Json axes = utils::Json::array();
    for (const char* axisName : {"z", "y", "x"}) {
        utils::Json axis{{"name", axisName}, {"type", "space"}};
        if (!options.voxelUnit.empty())
            axis["unit"] = options.voxelUnit;
        axes.push_back(std::move(axis));
    }
    multiscale["axes"] = std::move(axes);
    multiscale["datasets"] = utils::Json::array();
    for (size_t level = 0; level < options.numLevels; ++level) {
        double scaleZ = options.voxelSize;
        double scaleY = options.voxelSize;
        double scaleX = options.voxelSize;
        for (size_t i = 0; i < level; ++i) {
            scaleZ *= options.pyramid.downsampleZYX[0];
            scaleY *= options.pyramid.downsampleZYX[1];
            scaleX *= options.pyramid.downsampleZYX[2];
        }
        utils::Json scaleValues = utils::Json::array();
        scaleValues.push_back(scaleZ);
        scaleValues.push_back(scaleY);
        scaleValues.push_back(scaleX);
        utils::Json transforms = utils::Json::array();
        transforms.push_back(utils::Json{{"type", "scale"}, {"scale", std::move(scaleValues)}});
        multiscale["datasets"].push_back(utils::Json{
            {"path", std::to_string(level)},
            {"coordinateTransformations", std::move(transforms)}
        });
    }
    const char* downsamplingMethod = "mean";
    switch (options.pyramid.reduction) {
    case Volume::PyramidPolicy::Reduction::Mean:
        downsamplingMethod = "mean";
        break;
    case Volume::PyramidPolicy::Reduction::Max:
        downsamplingMethod = "max";
        break;
    case Volume::PyramidPolicy::Reduction::BinaryOr:
        downsamplingMethod = "binary_or";
        break;
    }
    multiscale["metadata"] = utils::Json{{"downsampling_method", downsamplingMethod}};
    attrs["multiscales"] = utils::Json::array();
    attrs["multiscales"].push_back(std::move(multiscale));
    writeTextFile(path / ".zattrs", attrs.dump(2) + "\n");
}

std::filesystem::path zarrArrayPathForLevel(const std::filesystem::path& root, int level)
{
    const auto levelPath = root / std::to_string(level);
    if (std::filesystem::exists(levelPath / ".zarray") ||
        std::filesystem::exists(levelPath / "zarr.json")) {
        return levelPath;
    }
    if (level == 0 &&
        (std::filesystem::exists(root / ".zarray") ||
         std::filesystem::exists(root / "zarr.json"))) {
        return root;
    }
    return levelPath;
}

bool hasAnyLocalZarrArray(const std::filesystem::path& path)
{
    if (std::filesystem::exists(path / ".zarray") ||
        std::filesystem::exists(path / "zarr.json")) {
        return true;
    }
    if (!std::filesystem::is_directory(path))
        return false;
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (!entry.is_directory())
            continue;
        if (std::filesystem::exists(entry.path() / ".zarray") ||
            std::filesystem::exists(entry.path() / "zarr.json")) {
            return true;
        }
    }
    return false;
}

Volume::PyramidPolicy::Reduction reductionFromMethod(const std::string& method)
{
    if (method == "max")
        return Volume::PyramidPolicy::Reduction::Max;
    if (method == "binary_or" || method == "or" || method == "nearest")
        return Volume::PyramidPolicy::Reduction::BinaryOr;
    return Volume::PyramidPolicy::Reduction::Mean;
}

} // namespace

Volume::Volume(std::filesystem::path path) : path_(std::move(path))
{
    loadMetadata();

    _width = metadata_["width"].get_int();
    _height = metadata_["height"].get_int();
    _slices = metadata_["slices"].get_int();

    zarrOpen();
}

Volume::Volume(std::filesystem::path path, RemoteConstructTag) : path_(std::move(path)) {}

Volume::~Volume() noexcept = default;

void Volume::loadMetadata()
{
    metadataAutoGenerated_ = false;

    auto metaPath = path_ / METADATA_FILE;
    if (std::filesystem::exists(metaPath)) {
        metadata_ = vc::json::load_json_file(metaPath);
    } else if (std::filesystem::exists(path_ / METADATA_FILE_ALT)) {
        auto altPath = path_ / METADATA_FILE_ALT;
        auto full = vc::json::load_json_file(altPath);
        if (!full.contains("scan")) {
            throw std::runtime_error(
                "metadata.json missing 'scan' key: " + altPath.string());
        }
        metadata_ = full;
        metadata_.update(full["scan"]);
        if (!metadata_.contains("format")) {
            metadata_["format"] = "zarr";
        }
        metaPath = altPath;
    } else {
        const auto baseName = path_.filename().string();
        metadata_["uuid"] = baseName;
        metadata_["name"] = baseName;
        metadata_["type"] = "vol";
        metadata_["format"] = "zarr";
        metadata_["width"] = 0;
        metadata_["height"] = 0;
        metadata_["slices"] = 0;
        metadata_["voxelsize"] = double{};
        metadata_["min"] = double{};
        metadata_["max"] = double{};
        metadataAutoGenerated_ = true;
        return;
    }
    vc::json::require_type(metadata_, "type", "vol", metaPath.string());
    vc::json::require_fields(metadata_, {"uuid", "width", "height", "slices"}, metaPath.string());
}

std::string Volume::id() const
{
    return metadata_["uuid"].get_string();
}

std::string Volume::name() const
{
    return metadata_["name"].get_string();
}

utils::Json Volume::rootAttributes() const
{
    if (isRemote())
        throw std::runtime_error("Volume::rootAttributes is only supported for local zarr volumes");
    const auto attrsPath = path_ / ".zattrs";
    if (!std::filesystem::exists(attrsPath))
        return utils::Json::object();
    return utils::Json::parse_file(attrsPath);
}

void Volume::writeRootAttributes(const utils::Json& attrs)
{
    if (isRemote())
        throw std::runtime_error("Volume::writeRootAttributes is only supported for local zarr volumes");
    if (!attrs.is_object())
        throw std::runtime_error("Volume::writeRootAttributes requires a JSON object");
    writeTextFile(path_ / ".zattrs", attrs.dump(2) + "\n");
}

void Volume::updateRootAttributes(const utils::Json& attrs)
{
    if (!attrs.is_object())
        throw std::runtime_error("Volume::updateRootAttributes requires a JSON object");
    auto merged = rootAttributes();
    merged.update(attrs);
    writeRootAttributes(merged);
}

void Volume::writeMetadata(const utils::Json& metadata)
{
    if (isRemote())
        throw std::runtime_error("Volume::writeMetadata is only supported for local zarr volumes");
    if (!metadata.is_object())
        throw std::runtime_error("Volume::writeMetadata requires a JSON object");
    writeTextFile(path_ / METADATA_FILE, metadata.dump(2) + "\n");
    metadata_ = metadata;
    if (metadata_.contains("width") && metadata_["width"].is_number())
        _width = metadata_["width"].get_int();
    if (metadata_.contains("height") && metadata_["height"].is_number())
        _height = metadata_["height"].get_int();
    if (metadata_.contains("slices") && metadata_["slices"].is_number())
        _slices = metadata_["slices"].get_int();
}

void Volume::updateMetadata(const utils::Json& metadata)
{
    if (!metadata.is_object())
        throw std::runtime_error("Volume::updateMetadata requires a JSON object");
    auto merged = metadata_;
    merged.update(metadata);
    writeMetadata(merged);
}

bool Volume::checkDir(const std::filesystem::path& path)
{
    return std::filesystem::is_directory(path) &&
           (std::filesystem::exists(path / METADATA_FILE) ||
            std::filesystem::exists(path / METADATA_FILE_ALT) ||
            hasAnyLocalZarrArray(path));
}

static int ceilDivPow2(int v, int level)
{
    const int64_t denom = int64_t{1} << level;
    return static_cast<int>((static_cast<int64_t>(v) + denom - 1) / denom);
}

void Volume::zarrOpen()
{
    if (!metadata_.contains("format") || metadata_["format"].get_string() != "zarr")
        return;

    auto opened = vc::render::openLocalZarrPyramid(path_);
    if (opened.shapes.empty()) {
        throw std::runtime_error("no physical zarr dataset directories found in " + path_.string());
    }
    zarrLevelShapes_ = opened.shapes;
    zarrLevelChunkShapes_ = opened.chunkShapes;
    zarrDtype_ = opened.dtype;
    zarrFillValue_ = opened.fillValue;

    try {
        const auto attrs = rootAttributes();
        if (attrs.contains("multiscales") && attrs["multiscales"].is_array() &&
            attrs["multiscales"].size() > 0) {
            const auto ms = attrs["multiscales"][0];
            if (ms.contains("metadata") && ms["metadata"].is_object()) {
                const auto meta = ms["metadata"];
                if (meta.contains("downsampling_method") &&
                    meta["downsampling_method"].is_string()) {
                    pyramidReduction_ = reductionFromMethod(
                        meta["downsampling_method"].get_string());
                }
            }
        }
    } catch (...) {
    }

    if (metadataAutoGenerated_) {
        bool hasReference = false;
        int baseSlices = 0;
        int baseHeight = 0;
        int baseWidth = 0;

        for (size_t level = 0; level < zarrLevelShapes_.size(); ++level) {
            const auto& shape = zarrLevelShapes_[level];
            if (shape[0] == 0 && shape[1] == 0 && shape[2] == 0) {
                continue;
            }
            const int levelInt = static_cast<int>(level);

            if (!hasReference) {
                const size_t scale = size_t{1} << levelInt;
                baseSlices = static_cast<int>(static_cast<size_t>(shape[0]) * scale);
                baseHeight = static_cast<int>(static_cast<size_t>(shape[1]) * scale);
                baseWidth = static_cast<int>(static_cast<size_t>(shape[2]) * scale);
                hasReference = true;
            }

            const int expectedSlices = ceilDivPow2(baseSlices, levelInt);
            const int expectedHeight = ceilDivPow2(baseHeight, levelInt);
            const int expectedWidth = ceilDivPow2(baseWidth, levelInt);

            constexpr int kMaxPerLevelPad = 128;
            auto padOK = [](long long actual, long long expected) {
                return actual >= expected && actual - expected < kMaxPerLevelPad;
            };
            if (!padOK(shape[0], expectedSlices) ||
                !padOK(shape[1], expectedHeight) ||
                !padOK(shape[2], expectedWidth)) {
                throw std::runtime_error(
                    "zarr level " + std::to_string(levelInt) + " shape [z,y,x]=("
                    + std::to_string(shape[0]) + ", " + std::to_string(shape[1]) + ", " + std::to_string(shape[2])
                    + ") does not match synthesized dimensions from first found scale (slices=" + std::to_string(baseSlices)
                    + ", height=" + std::to_string(baseHeight) + ", width=" + std::to_string(baseWidth)
                    + ") in " + path_.string());
            }
        }

        _slices = baseSlices;
        _height = baseHeight;
        _width = baseWidth;
        metadata_["slices"] = _slices;
        metadata_["height"] = _height;
        metadata_["width"] = _width;
    }

    // Verify each existing level shape against meta.json dimensions and level downscale.
    // zarr shape is [z, y, x] = [slices, height, width]
    if (!skipShapeCheck) {
        bool hasAnyPhysicalScale = false;
        for (size_t level = 0; level < zarrLevelShapes_.size(); ++level) {
            const auto& shape = zarrLevelShapes_[level];
            if (shape[0] == 0 && shape[1] == 0 && shape[2] == 0) {
                continue;
            }
            hasAnyPhysicalScale = true;

            const int expectedSlices = ceilDivPow2(_slices, static_cast<int>(level));
            const int expectedHeight = ceilDivPow2(_height, static_cast<int>(level));
            const int expectedWidth = ceilDivPow2(_width, static_cast<int>(level));

            constexpr int kMaxPerLevelPad = 128;
            auto padOK = [](long long actual, long long expected) {
                return actual >= expected && actual - expected < kMaxPerLevelPad;
            };
            if (!padOK(shape[0], expectedSlices) ||
                !padOK(shape[1], expectedHeight) ||
                !padOK(shape[2], expectedWidth)) {
                throw std::runtime_error(
                    "zarr level " + std::to_string(level) + " shape [z,y,x]=("
                    + std::to_string(shape[0]) + ", " + std::to_string(shape[1]) + ", " + std::to_string(shape[2])
                    + ") does not match expected downscaled meta.json dimensions (slices=" + std::to_string(expectedSlices)
                    + ", height=" + std::to_string(expectedHeight) + ", width=" + std::to_string(expectedWidth)
                    + ") in " + path_.string());
            }
        }
        if (!hasAnyPhysicalScale)
            throw std::runtime_error("no physical zarr dataset directories found in " + path_.string());
    }
}

std::shared_ptr<Volume> Volume::New(std::filesystem::path path)
{
    return std::make_shared<Volume>(std::move(path));
}

std::shared_ptr<Volume> Volume::New(std::filesystem::path path,
                                    const ZarrCreateOptions& options)
{
    if (options.overwriteExisting || !hasAnyLocalZarrArray(path)) {
        createLocalOmeZarr(path, options);
    }
    auto volume = std::make_shared<Volume>(std::move(path));
    volume->pyramidReduction_ = options.pyramid.reduction;
    return volume;
}

std::shared_ptr<Volume> Volume::NewFromUrl(
    const std::string& url,
    const std::filesystem::path& cacheRoot,
    const vc::HttpAuth& authIn)
{
    (void)cacheRoot;

    // Resolve s3:// URLs to https:// and detect AWS credentials
    auto resolved = vc::resolveRemoteUrl(url);
    vc::HttpAuth auth = authIn;
    if (resolved.useAwsSigv4 && auth.empty()) {
        auth = vc::loadAwsCredentials();
        if (auth.region.empty())
            auth.region = resolved.awsRegion;
        // SigV4 is implicitly enabled when access_key is non-empty.
        // If credentials are missing, clear them so the request proceeds
        // unsigned (anonymous access for public buckets).
        if (auth.access_key.empty() || auth.secret_key.empty())
            auth = {};  // anonymous — no SigV4
    } else if (resolved.useAwsSigv4 && auth.region.empty()) {
        auth.region = resolved.awsRegion;
    }

    const std::string remoteUrl = normalizeRemoteVolumeUrl(resolved.httpsUrl);

    vc::render::OpenedChunkedZarr opened;
    // Open the zarr metadata in memory. This performs the normal zarr metadata
    // reads, but does not stage .zarray/meta.json files on disk.
    // If stale AWS credentials are present, public buckets may reject the
    // signed request even though the same object is readable anonymously.
    try {
        opened = vc::render::openHttpZarrPyramid(remoteUrl, auth);
    } catch (const std::exception& e) {
        if (!resolved.useAwsSigv4 || auth.empty() || !isRemoteAuthError(e)) {
            throw;
        }

        vc::HttpAuth anonymousAuth;
        opened = vc::render::openHttpZarrPyramid(remoteUrl, anonymousAuth);
        auth = std::move(anonymousAuth);
    }

    if (opened.shapes.empty())
        throw std::runtime_error("No zarr levels found at " + remoteUrl);

    int firstPresentLevel = -1;
    for (std::size_t level = 0; level < opened.shapes.size(); ++level) {
        const auto& shape = opened.shapes[level];
        if (shape[0] != 0 || shape[1] != 0 || shape[2] != 0) {
            firstPresentLevel = static_cast<int>(level);
            break;
        }
    }
    if (firstPresentLevel < 0)
        throw std::runtime_error("No zarr levels found at " + remoteUrl);

    auto vol = std::make_shared<Volume>(std::filesystem::path{}, RemoteConstructTag{});

    vol->isRemote_ = true;
    vol->remoteUrl_ = remoteUrl;
    vol->remoteAuth_ = auth;
    vol->remoteNumScales_ = opened.shapes.size();
    vol->zarrLevelShapes_ = opened.shapes;
    vol->zarrLevelChunkShapes_ = opened.chunkShapes;
    vol->zarrDtype_ = opened.dtype;
    vol->zarrFillValue_ = opened.fillValue;
    const auto& firstShape = opened.shapes[static_cast<std::size_t>(firstPresentLevel)];
    const size_t firstScale = size_t{1} << firstPresentLevel;
    vol->_slices = static_cast<int>(static_cast<size_t>(firstShape[0]) * firstScale);
    vol->_height = static_cast<int>(static_cast<size_t>(firstShape[1]) * firstScale);
    vol->_width = static_cast<int>(static_cast<size_t>(firstShape[2]) * firstScale);

    const auto id = deriveRemoteVolumeId(remoteUrl);
    vol->metadata_["uuid"] = id;
    vol->metadata_["name"] = deriveRemoteVolumeName(remoteUrl);
    vol->metadata_["type"] = "vol";
    vol->metadata_["format"] = "zarr";
    vol->metadata_["width"] = vol->_width;
    vol->metadata_["height"] = vol->_height;
    vol->metadata_["slices"] = vol->_slices;
    vol->metadata_["voxelsize"] = double{};
    vol->metadata_["min"] = double{};
    vol->metadata_["max"] = double{};

    try {
        if (auto remoteMeta = loadRemoteVolumeMetadata(remoteUrl, auth)) {
            vol->metadata_.update(*remoteMeta);
            vol->metadata_["width"] = vol->_width;
            vol->metadata_["height"] = vol->_height;
            vol->metadata_["slices"] = vol->_slices;
        }
    } catch (const std::exception& e) {
        Logger()->warn("Failed to load remote volume metadata for '{}': {}", remoteUrl, e.what());
    }

    return vol;
}

int Volume::sliceWidth() const noexcept { return _width; }
int Volume::sliceHeight() const noexcept { return _height; }
int Volume::numSlices() const noexcept { return _slices; }
std::array<int, 3> Volume::shape() const noexcept { return {_slices, _height, _width}; }
std::array<int, 3> Volume::shape(int level) const
{
    if (level < 0) {
        throw std::out_of_range("Volume::shape level must be non-negative");
    }
    const auto index = static_cast<std::size_t>(level);
    if (index < zarrLevelShapes_.size()) {
        const auto shapeZYX = zarrLevelShapes_[index];
        if (shapeZYX[0] == 0 && shapeZYX[1] == 0 && shapeZYX[2] == 0) {
            throw std::out_of_range("Volume::shape level is not present");
        }
        return shapeZYX;
    }
    if (level == 0) {
        return shape();
    }
    throw std::out_of_range("Volume::shape level out of range");
}

std::array<int, 3> Volume::levelShape(int level) const
{
    return shape(level);
}

std::array<int, 3> Volume::chunkShape(int level) const
{
    if (level < 0) {
        throw std::out_of_range("Volume::chunkShape level must be non-negative");
    }
    const auto index = static_cast<std::size_t>(level);
    if (index >= zarrLevelChunkShapes_.size() || !hasScaleLevel(level)) {
        throw std::out_of_range("Volume::chunkShape requested missing zarr scale level " + std::to_string(level));
    }
    const auto chunkShapeZYX = zarrLevelChunkShapes_[index];
    if (chunkShapeZYX[0] <= 0 || chunkShapeZYX[1] <= 0 || chunkShapeZYX[2] <= 0) {
        throw std::runtime_error("Volume::chunkShape encountered invalid zarr chunk shape");
    }
    return chunkShapeZYX;
}

std::array<int, 3> Volume::chunkGridShape(int level) const
{
    const auto levelShapeZYX = shape(level);
    const auto chunkShapeZYX = chunkShape(level);
    return {
        (levelShapeZYX[0] + chunkShapeZYX[0] - 1) / chunkShapeZYX[0],
        (levelShapeZYX[1] + chunkShapeZYX[1] - 1) / chunkShapeZYX[1],
        (levelShapeZYX[2] + chunkShapeZYX[2] - 1) / chunkShapeZYX[2],
    };
}

size_t Volume::chunkCount(int level) const
{
    const auto grid = chunkGridShape(level);
    return static_cast<size_t>(grid[0]) *
           static_cast<size_t>(grid[1]) *
           static_cast<size_t>(grid[2]);
}

std::array<int, 3> Volume::shapeXyz() const noexcept { return {_width, _height, _slices}; }
double Volume::voxelSize() const
{
    return metadata_["voxelsize"].get_double();
}

size_t Volume::dtypeSize() const noexcept
{
    switch (zarrDtype_) {
    case vc::render::ChunkDtype::UInt8:
        return 1;
    case vc::render::ChunkDtype::UInt16:
        return 2;
    }
    return 1;
}

size_t Volume::numScales() const noexcept {
    if (isRemote_)
        return remoteNumScales_;
    return zarrLevelShapes_.size();
}

bool Volume::hasScaleLevel(int level) const noexcept
{
    if (level < 0)
        return false;
    const auto index = static_cast<std::size_t>(level);
    if (index >= zarrLevelShapes_.size())
        return false;
    const auto& shape = zarrLevelShapes_[index];
    return shape[0] != 0 || shape[1] != 0 || shape[2] != 0;
}

std::vector<int> Volume::presentScaleLevels() const
{
    std::vector<int> levels;
    for (std::size_t level = 0; level < zarrLevelShapes_.size(); ++level) {
        if (hasScaleLevel(static_cast<int>(level)))
            levels.push_back(static_cast<int>(level));
    }
    return levels;
}

int Volume::firstPresentScaleLevel() const
{
    for (std::size_t level = 0; level < zarrLevelShapes_.size(); ++level) {
        if (hasScaleLevel(static_cast<int>(level)))
            return static_cast<int>(level);
    }
    throw std::runtime_error("Volume has no present zarr scale levels");
}

int Volume::finestPresentScaleLevelAtOrBelow(int level) const
{
    if (level < 0)
        throw std::out_of_range("scale level must be non-negative");
    for (int candidate = level - 1; candidate >= 0; --candidate) {
        if (hasScaleLevel(candidate))
            return candidate;
    }
    throw std::out_of_range("no finer present zarr scale level available for virtual downsample");
}

// ============================================================================
// Cache management
// ============================================================================

vc::render::IChunkedArray* Volume::chunkedCache()
{
    std::lock_guard<std::mutex> lock(cacheMutex_);
    if (!chunkedCache_) {
        vc::render::ChunkCache::Options options;
        options.decodedByteCapacity = cacheBudgetHot_;
        options.maxConcurrentReads = ioThreads_ > 0 ? static_cast<std::size_t>(ioThreads_) : 16;
        chunkedCache_ = createChunkCache(std::move(options));
        if (!chunkedCache_) {
            throw std::runtime_error("Volume::chunkedCache failed to create chunk cache");
        }
    }
    return chunkedCache_.get();
}

std::shared_ptr<vc::render::ChunkCache> Volume::createChunkCache(
    vc::render::ChunkCache::Options options) const
{
    vc::render::OpenedChunkedZarr opened = isRemote_
        ? vc::render::openHttpZarrPyramid(remoteUrl_, remoteAuth_)
        : vc::render::openLocalZarrPyramid(path_);

    if (opened.fetchers.empty()) {
        return nullptr;
    }

    return std::make_shared<vc::render::ChunkCache>(
        makeChunkCacheLevelInfo(opened),
        std::move(opened.fetchers),
        opened.fillValue,
        opened.dtype,
        std::move(options));
}

void Volume::setCacheBudget(size_t hotBytes)
{
    std::lock_guard<std::mutex> lock(cacheMutex_);
    cacheBudgetHot_ = hotBytes;
    chunkedCache_.reset();
}

void Volume::setIOThreads(int count)
{
    std::lock_guard<std::mutex> lock(cacheMutex_);
    ioThreads_ = count;
    chunkedCache_.reset();
}

void Volume::invalidateCache()
{
    std::lock_guard<std::mutex> lock(cacheMutex_);
    chunkedCache_.reset();
}

// ============================================================================
// Sampling API
// ============================================================================

// Helper: apply optional post-processing from SampleParams in-place.
static void applyOptionalPostProcess(cv::Mat_<uint8_t>& img,
                                     const vc::SampleParams& params)
{
    if (!params.postProcess) return;
    vc::applyPostProcess(img, *params.postProcess);
}

// Helper: scale level-0 coords to pyramid level coords.
static const cv::Mat_<cv::Vec3f>& scaleCoords(const cv::Mat_<cv::Vec3f>& coords, int level)
{
    if (level <= 0) return coords;
    // Thread-local buffer avoids per-tile allocation for the scaled copy
    thread_local cv::Mat_<cv::Vec3f> scaled;
    float scale = 1.0f / static_cast<float>(1 << level);
    if (scaled.rows != coords.rows || scaled.cols != coords.cols) {
        scaled.create(coords.size());
    }
    const int total = coords.rows * coords.cols;
    const auto* src = coords.ptr<cv::Vec3f>();
    auto* dst = scaled.ptr<cv::Vec3f>();
    for (int i = 0; i < total; ++i) {
        dst[i] = src[i] * scale;
    }
    return scaled;
}

void Volume::sample(cv::Mat_<uint8_t>& out,
                    const cv::Mat_<cv::Vec3f>& coords,
                    const vc::SampleParams& params)
{
    if (coords.empty()) {
        out.release();
        return;
    }
    out.create(coords.size());
    const auto& scaled = scaleCoords(coords, params.level);
    readInterpolated3D(out, chunkedCache(), params.level, scaled, params.method);
    applyOptionalPostProcess(out, params);
}

void Volume::sample(cv::Mat_<uint16_t>& out,
                    const cv::Mat_<cv::Vec3f>& coords,
                    const vc::SampleParams& params)
{
    if (coords.empty()) {
        out.release();
        return;
    }
    out.create(coords.size());
    const auto& scaled = scaleCoords(coords, params.level);
    readInterpolated3D(out, chunkedCache(), params.level, scaled, params.method);
}

template <typename T>
static bool readVolumeZYXWithPolicy(Volume& volume,
                                    Array3D<T>& out,
                                    const std::array<int, 3>& offsetZYX,
                                    int level,
                                    Volume::MissingScaleLevelPolicy missingPolicy)
{
    if (level < 0)
        throw std::out_of_range("Volume::read level must be non-negative");

    auto* cache = volume.chunkedCache();
    if (volume.hasScaleLevel(level)) {
        readFromChunkedArrayZYX(out, offsetZYX, *cache, level);
        return true;
    }

    switch (missingPolicy) {
    case Volume::MissingScaleLevelPolicy::Error:
        throw std::out_of_range("Volume::read requested missing zarr scale level " + std::to_string(level));
    case Volume::MissingScaleLevelPolicy::AllFill:
        fillFromChunkedArrayFillValue(out, *cache);
        return true;
    case Volume::MissingScaleLevelPolicy::Empty:
        return false;
    case Volume::MissingScaleLevelPolicy::VirtualDownsample:
        break;
    }

    const int sourceLevel = volume.finestPresentScaleLevelAtOrBelow(level);
    const int levelDelta = level - sourceLevel;
    if (levelDelta <= 0 || levelDelta >= static_cast<int>(sizeof(int) * 8 - 1)) {
        throw std::out_of_range("invalid virtual zarr scale level delta");
    }
    const int factor = 1 << levelDelta;
    const auto outShape = out.shape();
    if (outShape[0] == 0 || outShape[1] == 0 || outShape[2] == 0)
        return true;

    std::array<size_t, 3> sourceReadShape{
        outShape[0] * static_cast<size_t>(factor),
        outShape[1] * static_cast<size_t>(factor),
        outShape[2] * static_cast<size_t>(factor),
    };
    Array3D<T> source(sourceReadShape);
    std::array<int, 3> sourceOffset{
        offsetZYX[0] * factor,
        offsetZYX[1] * factor,
        offsetZYX[2] * factor,
    };
    readFromChunkedArrayZYX(source, sourceOffset, *cache, sourceLevel);
    downsampleMeanZYX(out, source, sourceOffset, cache->shape(sourceLevel), factor);
    return true;
}

bool Volume::readZYX(Array3D<uint8_t>& out,
                           const std::array<int, 3>& offsetZYX,
                           int level,
                           MissingScaleLevelPolicy missingPolicy)
{
    return readVolumeZYXWithPolicy(*this, out, offsetZYX, level, missingPolicy);
}

bool Volume::readZYX(Array3D<uint16_t>& out,
                           const std::array<int, 3>& offsetZYX,
                           int level,
                           MissingScaleLevelPolicy missingPolicy)
{
    return readVolumeZYXWithPolicy(*this, out, offsetZYX, level, missingPolicy);
}

void Volume::readZYX(Array3D<uint8_t>& out,
                           const std::array<int, 3>& offsetZYX,
                           vc::render::IChunkedArray& array,
                           int level)
{
    readFromChunkedArrayZYX(out, offsetZYX, array, level);
}

void Volume::readZYX(Array3D<uint16_t>& out,
                           const std::array<int, 3>& offsetZYX,
                           vc::render::IChunkedArray& array,
                           int level)
{
    readFromChunkedArrayZYX(out, offsetZYX, array, level);
}

bool Volume::readXYZ(Array3D<uint8_t>& out,
                           const std::array<int, 3>& offsetXYZ,
                           int level,
                           MissingScaleLevelPolicy missingPolicy)
{
    return readZYX(out, xyzToZyx(offsetXYZ), level, missingPolicy);
}

bool Volume::readXYZ(Array3D<uint16_t>& out,
                           const std::array<int, 3>& offsetXYZ,
                           int level,
                           MissingScaleLevelPolicy missingPolicy)
{
    return readZYX(out, xyzToZyx(offsetXYZ), level, missingPolicy);
}

template <typename T>
static void writeVolumeZYX(Volume& volume,
                           const Array3D<T>& data,
                           const std::array<int, 3>& offsetZYX,
                           int level)
{
    if (volume.isRemote())
        throw std::runtime_error("Volume::write is only supported for local zarr volumes");
    if (level < 0)
        throw std::out_of_range("Volume::write level must be non-negative");
    if (!volume.hasScaleLevel(level))
        throw std::out_of_range("Volume::write requested missing zarr scale level " + std::to_string(level));
    if (volume.dtype() != chunkDtypeFor<T>())
        throw std::runtime_error("Volume::write dtype does not match volume dtype");

    std::array<size_t, 3> writeOffset{};
    for (size_t d = 0; d < 3; ++d) {
        if (offsetZYX[d] < 0)
            throw std::out_of_range("Volume::write offset must be non-negative");
        writeOffset[d] = static_cast<size_t>(offsetZYX[d]);
    }

    auto array = openLocalZarrArrayForWrite(zarrArrayPathForLevel(volume.path(), level));
    writeZarrRegionZYX(array, data, writeOffset);

    std::array<size_t, 3> affectedOffset = writeOffset;
    std::array<size_t, 3> affectedShape = data.shape();
    for (int dstLevel = level + 1;
         dstLevel < static_cast<int>(volume.numScales()) && volume.hasScaleLevel(dstLevel);
         ++dstLevel) {
        const int srcLevel = dstLevel - 1;
        auto srcArray = openLocalZarrArrayForWrite(zarrArrayPathForLevel(volume.path(), srcLevel));
        auto dstArray = openLocalZarrArrayForWrite(zarrArrayPathForLevel(volume.path(), dstLevel));
        const auto srcVolumeShapeInt = volume.shape(srcLevel);
        const auto dstVolumeShapeInt = volume.shape(dstLevel);
        std::array<size_t, 3> srcVolumeShape{
            static_cast<size_t>(srcVolumeShapeInt[0]),
            static_cast<size_t>(srcVolumeShapeInt[1]),
            static_cast<size_t>(srcVolumeShapeInt[2]),
        };
        std::array<size_t, 3> dstVolumeShape{
            static_cast<size_t>(dstVolumeShapeInt[0]),
            static_cast<size_t>(dstVolumeShapeInt[1]),
            static_cast<size_t>(dstVolumeShapeInt[2]),
        };
        std::array<double, 3> scaleZYX{};
        for (size_t d = 0; d < 3; ++d) {
            scaleZYX[d] = static_cast<double>(srcVolumeShape[d]) /
                          static_cast<double>(std::max<size_t>(1, dstVolumeShape[d]));
            if (scaleZYX[d] < 1.0)
                scaleZYX[d] = 1.0;
        }

        std::array<size_t, 3> dstOffset{};
        std::array<size_t, 3> dstEnd{};
        std::array<size_t, 3> dstShape{};
        for (size_t d = 0; d < 3; ++d) {
            const size_t srcEnd = affectedOffset[d] + affectedShape[d];
            dstOffset[d] = std::min(
                dstVolumeShape[d],
                static_cast<size_t>(std::floor(static_cast<double>(affectedOffset[d]) / scaleZYX[d])));
            dstEnd[d] = std::min(
                dstVolumeShape[d],
                static_cast<size_t>(std::ceil(static_cast<double>(srcEnd) / scaleZYX[d])));
            dstShape[d] = dstEnd[d] > dstOffset[d] ? dstEnd[d] - dstOffset[d] : 0;
        }
        if (dstShape[0] == 0 || dstShape[1] == 0 || dstShape[2] == 0)
            break;

        std::array<size_t, 3> srcReadOffset{
            std::min(srcVolumeShape[0], static_cast<size_t>(std::floor(static_cast<double>(dstOffset[0]) * scaleZYX[0]))),
            std::min(srcVolumeShape[1], static_cast<size_t>(std::floor(static_cast<double>(dstOffset[1]) * scaleZYX[1]))),
            std::min(srcVolumeShape[2], static_cast<size_t>(std::floor(static_cast<double>(dstOffset[2]) * scaleZYX[2]))),
        };
        std::array<size_t, 3> srcReadEnd{
            std::min(srcVolumeShape[0], static_cast<size_t>(std::ceil(static_cast<double>(dstEnd[0]) * scaleZYX[0]))),
            std::min(srcVolumeShape[1], static_cast<size_t>(std::ceil(static_cast<double>(dstEnd[1]) * scaleZYX[1]))),
            std::min(srcVolumeShape[2], static_cast<size_t>(std::ceil(static_cast<double>(dstEnd[2]) * scaleZYX[2]))),
        };
        std::array<size_t, 3> srcReadShape{
            srcReadEnd[0] > srcReadOffset[0] ? srcReadEnd[0] - srcReadOffset[0] : 0,
            srcReadEnd[1] > srcReadOffset[1] ? srcReadEnd[1] - srcReadOffset[1] : 0,
            srcReadEnd[2] > srcReadOffset[2] ? srcReadEnd[2] - srcReadOffset[2] : 0,
        };
        if (srcReadShape[0] == 0 || srcReadShape[1] == 0 || srcReadShape[2] == 0)
            break;

        Array3D<T> source(srcReadShape);
        readZarrRegionZYX(srcArray, source, srcReadOffset);

        Array3D<T> downsampled(dstShape);
        downsampleZYX(downsampled,
                      source,
                      srcReadOffset,
                      srcVolumeShape,
                      scaleZYX,
                      volume.pyramidReduction());
        writeZarrRegionZYX(dstArray, downsampled, dstOffset);

        affectedOffset = dstOffset;
        affectedShape = dstShape;
    }

    volume.invalidateCache();
}

void Volume::writeZYX(const Array3D<uint8_t>& data,
                      const std::array<int, 3>& offsetZYX,
                      int level)
{
    writeVolumeZYX(*this, data, offsetZYX, level);
}

void Volume::writeZYX(const Array3D<uint16_t>& data,
                      const std::array<int, 3>& offsetZYX,
                      int level)
{
    writeVolumeZYX(*this, data, offsetZYX, level);
}

void Volume::writeXYZ(const Array3D<uint8_t>& data,
                      const std::array<int, 3>& offsetXYZ,
                      int level)
{
    writeZYX(data, xyzToZyx(offsetXYZ), level);
}

void Volume::writeXYZ(const Array3D<uint16_t>& data,
                      const std::array<int, 3>& offsetXYZ,
                      int level)
{
    writeZYX(data, xyzToZyx(offsetXYZ), level);
}

std::optional<std::vector<std::byte>> Volume::readChunk(
    int level,
    const std::array<size_t, 3>& chunkZYX) const
{
    if (isRemote())
        throw std::runtime_error("Volume::readChunk is only supported for local zarr volumes");
    if (level < 0)
        throw std::out_of_range("Volume::readChunk level must be non-negative");
    if (!hasScaleLevel(level))
        throw std::out_of_range("Volume::readChunk requested missing zarr scale level " + std::to_string(level));

    auto array = openLocalZarrArrayForWrite(zarrArrayPathForLevel(path(), level));
    return array.read_chunk(chunkZYX);
}

std::vector<std::byte> Volume::readChunkOrFill(
    int level,
    const std::array<size_t, 3>& chunkZYX) const
{
    if (auto chunk = readChunk(level, chunkZYX))
        return std::move(*chunk);

    auto array = openLocalZarrArrayForWrite(zarrArrayPathForLevel(path(), level));
    return filledChunkBytes(array);
}

bool Volume::chunkExists(
    int level,
    const std::array<size_t, 3>& chunkZYX) const
{
    if (isRemote())
        throw std::runtime_error("Volume::chunkExists is only supported for local zarr volumes");
    if (level < 0)
        throw std::out_of_range("Volume::chunkExists level must be non-negative");
    if (!hasScaleLevel(level))
        throw std::out_of_range("Volume::chunkExists requested missing zarr scale level " + std::to_string(level));

    auto array = openLocalZarrArrayForWrite(zarrArrayPathForLevel(path(), level));
    if (array.is_sharded())
        return array.inner_chunk_exists(chunkZYX);
    return array.chunk_exists(chunkZYX);
}

void Volume::writeChunk(int level,
                        const std::array<size_t, 3>& chunkZYX,
                        std::span<const std::byte> data)
{
    writeChunk(level, chunkZYX, data, ChunkWriteOptions{});
}

void Volume::writeChunk(int level,
                        const std::array<size_t, 3>& chunkZYX,
                        std::span<const std::byte> data,
                        ChunkWriteOptions options)
{
    if (isRemote())
        throw std::runtime_error("Volume::writeChunk is only supported for local zarr volumes");
    if (level < 0)
        throw std::out_of_range("Volume::writeChunk level must be non-negative");
    if (!hasScaleLevel(level))
        throw std::out_of_range("Volume::writeChunk requested missing zarr scale level " + std::to_string(level));

    auto array = openLocalZarrArrayForWrite(zarrArrayPathForLevel(path(), level));
    const size_t expectedBytes = array.metadata().sub_chunk_byte_size();
    if (data.size() != expectedBytes) {
        throw std::runtime_error("Volume::writeChunk byte count does not match zarr chunk byte size");
    }

    if (!options.writeEmptyChunks && chunkMatchesFill(array, data)) {
        removeChunk(level, chunkZYX);
        return;
    }

    if (array.is_sharded())
        array.write_inner_chunk_to_shard(chunkZYX, data);
    else
        array.write_chunk(chunkZYX, data);
    invalidateCache();
}

bool Volume::removeChunk(int level,
                         const std::array<size_t, 3>& chunkZYX)
{
    if (isRemote())
        throw std::runtime_error("Volume::removeChunk is only supported for local zarr volumes");
    if (level < 0)
        throw std::out_of_range("Volume::removeChunk level must be non-negative");
    if (!hasScaleLevel(level))
        throw std::out_of_range("Volume::removeChunk requested missing zarr scale level " + std::to_string(level));

    auto array = openLocalZarrArrayForWrite(zarrArrayPathForLevel(path(), level));
    if (array.is_sharded()) {
        const bool existed = array.inner_chunk_exists(chunkZYX);
        array.mark_inner_chunk_empty(chunkZYX);
        invalidateCache();
        return existed;
    }

    const auto chunkPath = array.chunk_path(chunkZYX);
    std::error_code ec;
    const bool removed = std::filesystem::remove(chunkPath, ec);
    if (ec)
        throw std::runtime_error("failed removing zarr chunk: " + chunkPath.string());
    if (removed)
        invalidateCache();
    return removed;
}
