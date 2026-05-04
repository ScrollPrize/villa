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

    const T fill = typedFillValue<T>(array);
    const size_t chunkStrideY = static_cast<size_t>(chunkShape[2]);
    const size_t chunkStrideZ = static_cast<size_t>(chunkShape[1]) * chunkStrideY;

    #pragma omp parallel
    {
        struct CachedChunk {
            int cz = std::numeric_limits<int>::min();
            int cy = std::numeric_limits<int>::min();
            int cx = std::numeric_limits<int>::min();
            bool allFill = false;
            const T* data = nullptr;
            std::shared_ptr<const std::vector<std::byte>> bytes;
        } cached;

        auto loadChunk = [&](int cz, int cy, int cx) {
            if (cached.cz == cz && cached.cy == cy && cached.cx == cx) {
                return;
            }
            cached = {};
            cached.cz = cz;
            cached.cy = cy;
            cached.cx = cx;

            const auto result = array.getChunkBlocking(level, cz, cy, cx);
            if (result.status == ChunkStatus::AllFill) {
                cached.allFill = true;
            } else if (result.status == ChunkStatus::Data && result.bytes) {
                cached.bytes = result.bytes;
                cached.data = reinterpret_cast<const T*>(cached.bytes->data());
            } else if (result.status == ChunkStatus::Error) {
                throw std::runtime_error(result.error.empty() ? "chunk fetch failed" : result.error);
            } else {
                cached.allFill = true;
            }
        };

        #pragma omp for schedule(dynamic, 4) collapse(2)
        for (size_t z = 0; z < outShape[0]; ++z) {
            for (size_t y = 0; y < outShape[1]; ++y) {
                const int iz = z0 + static_cast<int>(z);
                const int iy = y0 + static_cast<int>(y);
                if (iz < 0 || iz >= volumeShape[0] ||
                    iy < 0 || iy >= volumeShape[1]) {
                    continue;
                }
                const int cz = iz / chunkShape[0];
                const int cy = iy / chunkShape[1];
                const int lz = iz - cz * chunkShape[0];
                const int ly = iy - cy * chunkShape[1];
                for (size_t x = 0; x < outShape[2]; ++x) {
                    const int ix = x0 + static_cast<int>(x);
                    if (ix < 0 || ix >= volumeShape[2]) {
                        continue;
                    }
                    const int cx = ix / chunkShape[2];
                    const int lx = ix - cx * chunkShape[2];
                    loadChunk(cz, cy, cx);
                    if (cached.allFill || !cached.data) {
                        out(z, y, x) = fill;
                    } else {
                        const size_t offset = static_cast<size_t>(lz) * chunkStrideZ +
                                              static_cast<size_t>(ly) * chunkStrideY +
                                              static_cast<size_t>(lx);
                        out(z, y, x) = cached.data[offset];
                    }
                }
            }
        }
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

std::array<size_t, 3> ceilHalfShape(const std::array<size_t, 3>& shape)
{
    return {(shape[0] + 1) / 2, (shape[1] + 1) / 2, (shape[2] + 1) / 2};
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
        levelShape = ceilHalfShape(levelShape);
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
        const double scale = options.voxelSize * static_cast<double>(size_t{1} << level);
        utils::Json scaleValues = utils::Json::array();
        scaleValues.push_back(scale);
        scaleValues.push_back(scale);
        scaleValues.push_back(scale);
        utils::Json transforms = utils::Json::array();
        transforms.push_back(utils::Json{{"type", "scale"}, {"scale", std::move(scaleValues)}});
        multiscale["datasets"].push_back(utils::Json{
            {"path", std::to_string(level)},
            {"coordinateTransformations", std::move(transforms)}
        });
    }
    multiscale["metadata"] = utils::Json{{"downsampling_method", "mean"}};
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

bool Volume::checkDir(const std::filesystem::path& path)
{
    return std::filesystem::is_directory(path) &&
           (std::filesystem::exists(path / METADATA_FILE) ||
            std::filesystem::exists(path / METADATA_FILE_ALT) ||
            std::filesystem::exists(path / ".zgroup") ||
            std::filesystem::exists(path / ".zattrs"));
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
    zarrDtype_ = opened.dtype;

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
    if (options.overwriteExisting || !checkDir(path)) {
        createLocalOmeZarr(path, options);
    }
    return std::make_shared<Volume>(std::move(path));
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
    vol->zarrDtype_ = opened.dtype;
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
std::array<int, 3> Volume::shapeXyz() const noexcept { return {_width, _height, _slices}; }
double Volume::voxelSize() const
{
    return metadata_["voxelsize"].get_double();
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
    const auto& scaled = scaleCoords(coords, params.level);
    readInterpolated3D(out, chunkedCache(), params.level, scaled, params.method);
    applyOptionalPostProcess(out, params);
}

void Volume::sample(cv::Mat_<uint16_t>& out,
                    const cv::Mat_<cv::Vec3f>& coords,
                    const vc::SampleParams& params)
{
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

        std::array<size_t, 3> dstOffset{};
        std::array<size_t, 3> dstEnd{};
        std::array<size_t, 3> dstShape{};
        for (size_t d = 0; d < 3; ++d) {
            const size_t srcEnd = affectedOffset[d] + affectedShape[d];
            dstOffset[d] = affectedOffset[d] / 2;
            dstEnd[d] = (srcEnd + 1) / 2;
            dstShape[d] = dstEnd[d] - dstOffset[d];
        }
        if (dstShape[0] == 0 || dstShape[1] == 0 || dstShape[2] == 0)
            break;

        const auto srcVolumeShape = volume.shape(srcLevel);
        std::array<size_t, 3> srcReadOffset{
            dstOffset[0] * 2,
            dstOffset[1] * 2,
            dstOffset[2] * 2,
        };
        std::array<size_t, 3> srcReadShape{
            std::min(dstShape[0] * 2, static_cast<size_t>(srcVolumeShape[0]) - srcReadOffset[0]),
            std::min(dstShape[1] * 2, static_cast<size_t>(srcVolumeShape[1]) - srcReadOffset[1]),
            std::min(dstShape[2] * 2, static_cast<size_t>(srcVolumeShape[2]) - srcReadOffset[2]),
        };

        Array3D<T> source(srcReadShape);
        readZarrRegionZYX(srcArray, source, srcReadOffset);

        Array3D<T> downsampled(dstShape);
        std::array<int, 3> srcReadOffsetInt{
            static_cast<int>(srcReadOffset[0]),
            static_cast<int>(srcReadOffset[1]),
            static_cast<int>(srcReadOffset[2]),
        };
        downsampleMeanZYX(downsampled, source, srcReadOffsetInt, srcVolumeShape, 2);
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
