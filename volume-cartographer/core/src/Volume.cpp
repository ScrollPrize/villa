#include "vc/core/types/Volume.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
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
#include "vc/core/types/VcDataset.hpp"
#include "vc/core/render/ZarrChunkFetcher.hpp"
#include "vc/core/render/PersistentZarrCacheBudget.hpp"
#include "vc/core/util/HttpFetch.hpp"
#include "vc/core/util/RemoteUrl.hpp"
#include "vc/core/util/RemoteFileCache.hpp"
#include "vc/core/util/RemoteMetadataCache.hpp"
#include "vc/core/util/RemoteCacheSettings.hpp"
#include "vc/core/util/VoxelSizeMetadata.hpp"
#include "vc/core/render/IChunkedArray.hpp"
#include "vc/core/render/ChunkFetch.hpp"
#include "utils/hash.hpp"
#include "utils/zarr.hpp"

static const std::filesystem::path METADATA_FILE = "meta.json";
static const std::filesystem::path METADATA_FILE_ALT = "metadata.json";

namespace
{

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

// Record the voxel size resolved from the store's document. When there is
// none, say so: a volume without one reports 0, and every physical measurement
// derived from it -- surface areas, the min_area_cm gate, the scale bar --
// silently becomes 0 too (#1603).
void applyResolvedVoxelSize(utils::Json& metadata,
                            std::optional<double> resolved,
                            const std::string& description)
{
    if (resolved) {
        metadata["voxelsize"] = *resolved;
        return;
    }
    if (vc::json::number_or(metadata, "voxelsize", 0.0) > 0.0) {
        return;
    }
    Logger()->warn(
        "volume '{}': no voxel size in its store metadata, so physical measurements "
        "are unavailable for it",
        description);
}

std::optional<utils::Json> loadRemoteVolumeMetadata(const std::string& remoteUrl,
                                                    const vc::HttpAuth& auth)
{
    const auto normalize = [&](utils::Json json, const std::string& url) -> utils::Json {
        if (!json.is_object()) {
            throw std::runtime_error("remote volume metadata is not an object: " + url);
        }
        // Resolve from the document as published, before `scan` is flattened
        // into the root.
        const auto voxelSize = vc::metadata::voxelSizeFromStoreMetadata(json);
        if (json.contains("scan") && json["scan"].is_object()) {
            json.update(json["scan"]);
        }
        if (!json.contains("format")) {
            json["format"] = "zarr";
        }
        applyResolvedVoxelSize(json, voxelSize, url);
        return json;
    };

    const auto loadUrl = [&](const std::string& url) -> std::optional<utils::Json> {
        auto& cache = vc::core::util::remoteMetadataCache();
        std::string body;
        if (const auto answer = cache.lookup(url)) {
            if (*answer)
                body.assign(reinterpret_cast<const char*>((*answer)->data()), (*answer)->size());
        } else {
            body = vc::httpGetString(url, auth);
            if (vc::core::util::RemoteMetadataCache::cacheable(url)) {
                std::vector<std::byte> bytes(body.size());
                std::memcpy(bytes.data(), body.data(), body.size());
                cache.record(url, body.empty()
                    ? vc::core::util::RemoteMetadataCache::Answer{}
                    : vc::core::util::RemoteMetadataCache::Answer{std::move(bytes)});
            }
        }
        if (body.empty()) {
            return std::nullopt;
        }
        return normalize(utils::Json::parse(body), url);
    };

    if (auto meta = loadUrl(vc::joinRemoteUrlPath(remoteUrl, METADATA_FILE.string()))) {
        return meta;
    }
    return loadUrl(vc::joinRemoteUrlPath(remoteUrl, METADATA_FILE_ALT.string()));
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

std::string normalizeLocalVolumePathForId(const std::filesystem::path& path)
{
    return std::filesystem::absolute(path).lexically_normal().string();
}

std::string deriveLocalVolumeName(const std::filesystem::path& path)
{
    const auto filename = path.filename().string();
    if (!filename.empty() &&
        std::all_of(filename.begin(), filename.end(), [](unsigned char c) {
            return std::isdigit(c) != 0;
        })) {
        const auto parentName = path.parent_path().filename().string();
        if (!parentName.empty())
            return parentName;
    }
    return filename.empty() ? std::string("volume") : filename;
}

std::string deriveLocalVolumeId(const std::filesystem::path& path)
{
    const std::string name = deriveLocalVolumeName(path);
    const std::string normalized = normalizeLocalVolumePathForId(path);
    const auto hash = utils::fnv1a(std::string_view(normalized));
    std::ostringstream out;
    out << name << "-" << std::hex << std::nouppercase << std::setw(16)
        << std::setfill('0') << hash;
    return out.str();
}

std::string deriveRebasedRemoteCacheId(const std::string& descriptiveId,
                                       const std::string& sourceUrl,
                                       int baseScaleLevel)
{
    const std::string identity = sourceUrl + "#vc-base-scale=" +
                                 std::to_string(baseScaleLevel);
    const auto hash = utils::fnv1a(std::string_view(identity));
    std::ostringstream out;
    out << descriptiveId << "-vc-base-" << baseScaleLevel << "-"
        << std::hex << std::nouppercase << std::setw(16) << std::setfill('0') << hash;
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
    std::atomic<bool> chunkReadFailed{false};
    ChunkKey failedChunk{};
    std::string chunkReadError;
    std::mutex chunkReadErrorMutex;

    auto recordChunkReadError = [&](int cz, int cy, int cx, const std::string& error) {
        bool expected = false;
        if (!chunkReadFailed.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            return;
        }

        std::lock_guard<std::mutex> lock(chunkReadErrorMutex);
        failedChunk = ChunkKey{level, cz, cy, cx};
        chunkReadError = error.empty() ? "chunk fetch failed" : error;
    };

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

        auto loadChunk = [&](int cz, int cy, int cx) -> bool {
            if (chunkReadFailed.load(std::memory_order_acquire)) {
                return false;
            }
            if (cached.cz == cz && cached.cy == cy && cached.cx == cx) {
                return true;
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
                recordChunkReadError(cz, cy, cx, result.error);
                return false;
            } else {
                cached.allFill = true;
            }
            return true;
        };

        #pragma omp for schedule(dynamic, 4) collapse(2)
        for (size_t z = 0; z < outShape[0]; ++z) {
            for (size_t y = 0; y < outShape[1]; ++y) {
                if (chunkReadFailed.load(std::memory_order_acquire)) {
                    continue;
                }
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
                    if (chunkReadFailed.load(std::memory_order_acquire)) {
                        break;
                    }
                    const int ix = x0 + static_cast<int>(x);
                    if (ix < 0 || ix >= volumeShape[2]) {
                        continue;
                    }
                    const int cx = ix / chunkShape[2];
                    const int lx = ix - cx * chunkShape[2];
                    if (!loadChunk(cz, cy, cx)) {
                        break;
                    }
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

    if (chunkReadFailed.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(chunkReadErrorMutex);
        std::ostringstream oss;
        oss << "Volume::read failed fetching chunk "
            << failedChunk.level << "/" << failedChunk.iz << "/" << failedChunk.iy << "/"
            << failedChunk.ix << ": " << chunkReadError;
        throw std::runtime_error(oss.str());
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

template <typename T>
T typedFillValue(double fill)
{
    if constexpr (std::is_same_v<T, uint8_t>)
        return static_cast<uint8_t>(std::clamp(fill, 0.0, 255.0));
    else
        return static_cast<uint16_t>(std::clamp(fill, 0.0, 65535.0));
}

// Opens with the codec registry that backs the chunked read cache
// (ZarrChunkFetcher / VcDataset).
utils::ZarrArray openLocalZarrArrayForRead(const std::filesystem::path& path,
                                           int dtypeSize)
{
    return utils::ZarrArray::open(path, vc::buildZarrCodecRegistry(dtypeSize));
}

template <typename T>
void fillRawChunk(std::vector<std::byte>& chunkBytes, T fill)
{
    auto* typed = reinterpret_cast<T*>(chunkBytes.data());
    std::fill(typed, typed + chunkBytes.size() / sizeof(T), fill);
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
        // Resolve from the document as published, before `scan` is flattened
        // into the root.
        const auto voxelSize = vc::metadata::voxelSizeFromStoreMetadata(full);
        metadata_ = full;
        metadata_.update(full["scan"]);
        if (!metadata_.contains("format")) {
            metadata_["format"] = "zarr";
        }
        applyResolvedVoxelSize(metadata_, voxelSize, altPath.string());
        metaPath = altPath;
    } else {
        metadata_["uuid"] = deriveLocalVolumeId(path_);
        metadata_["name"] = deriveLocalVolumeName(path_);
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

static int ceilDivPow2(int v, int level)
{
    const int64_t denom = int64_t{1} << level;
    return static_cast<int>((static_cast<int64_t>(v) + denom - 1) / denom);
}

static bool paddedShapeOK(long long actual, long long expected, int padMultiple)
{
    const long long maxPad = std::max(1, padMultiple);
    return actual >= expected && actual - expected < maxPad;
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
    zarrLevelStorageChunkShapes_ = opened.storageChunkShapes;
    zarrDtype_ = opened.dtype;
    zarrFillValue_ = opened.fillValue;

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

            const auto storageChunkShape = level < zarrLevelStorageChunkShapes_.size()
                ? zarrLevelStorageChunkShapes_[level]
                : std::array<int, 3>{128, 128, 128};
            if (!paddedShapeOK(shape[0], expectedSlices, storageChunkShape[0]) ||
                !paddedShapeOK(shape[1], expectedHeight, storageChunkShape[1]) ||
                !paddedShapeOK(shape[2], expectedWidth, storageChunkShape[2])) {
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

            const auto storageChunkShape = level < zarrLevelStorageChunkShapes_.size()
                ? zarrLevelStorageChunkShapes_[level]
                : std::array<int, 3>{128, 128, 128};
            if (!paddedShapeOK(shape[0], expectedSlices, storageChunkShape[0]) ||
                !paddedShapeOK(shape[1], expectedHeight, storageChunkShape[1]) ||
                !paddedShapeOK(shape[2], expectedWidth, storageChunkShape[2])) {
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
                                    const utils::Json& metadata)
{
    auto volume = std::make_shared<Volume>(std::move(path));
    if (metadata.is_object()) {
        if (metadata.contains("voxelsize") && metadata["voxelsize"].is_number()) {
            const double value = metadata["voxelsize"].get_double();
            volume->hasExplicitVoxelSizeOverride_ = std::isfinite(value) && value > 0.0;
        }
        volume->metadata_.update(metadata);
        volume->metadata_["width"] = volume->_width;
        volume->metadata_["height"] = volume->_height;
        volume->metadata_["slices"] = volume->_slices;
    }
    return volume;
}

std::shared_ptr<Volume> Volume::NewFromUrl(
    const std::string& url,
    const vc::HttpAuth& authIn,
    const utils::Json& metadata,
    bool discoverAwsCredentials)
{
    vc::render::RemoteZarrOpenOptions openOptions;
    openOptions.auth = authIn;
    openOptions.discoverAwsCredentials = discoverAwsCredentials;
    auto remoteOpen = vc::render::openRemoteZarrPyramid(url, std::move(openOptions));
    auto opened = std::move(remoteOpen.opened);
    auto auth = std::move(remoteOpen.auth);
    const auto& spec = remoteOpen.spec;
    const std::string& remoteUrl = spec.sourceUrl;

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
    vol->remoteLocator_ = spec.portableLocator;
    vol->baseScaleLevel_ = spec.baseScaleLevel;
    vol->remoteAuth_ = auth;
    vol->remoteCacheRoot_ = vc::settings::remoteCachePath();
    vol->remoteNumScales_ = opened.shapes.size();
    vol->zarrLevelShapes_ = opened.shapes;
    vol->zarrLevelChunkShapes_ = opened.chunkShapes;
    vol->zarrLevelStorageChunkShapes_ = opened.storageChunkShapes;
    vol->zarrDtype_ = opened.dtype;
    vol->zarrFillValue_ = opened.fillValue;
    const auto& firstShape = opened.shapes[static_cast<std::size_t>(firstPresentLevel)];
    const size_t firstScale = spec.baseScaleLevel > 0 ? 1 : size_t{1} << firstPresentLevel;
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

    std::optional<double> nativeRemoteVoxelSize;
    try {
        if (auto remoteMeta = loadRemoteVolumeMetadata(remoteUrl, auth)) {
            if (remoteMeta->contains("voxelsize") &&
                (*remoteMeta)["voxelsize"].is_number()) {
                const double value = (*remoteMeta)["voxelsize"].get_double();
                if (std::isfinite(value) && value > 0.0)
                    nativeRemoteVoxelSize = value;
            }
            vol->metadata_.update(*remoteMeta);
            vol->metadata_["width"] = vol->_width;
            vol->metadata_["height"] = vol->_height;
            vol->metadata_["slices"] = vol->_slices;
        }
    } catch (const std::exception& e) {
        Logger()->warn("Failed to load remote volume metadata for '{}': {}", remoteUrl, e.what());
    }

    // Preserve the historical source cache name before project metadata can
    // replace the annotation UUID (for example, with an open-data catalog ID).
    // Upstream metadata UUIDs still name their existing caches; otherwise the
    // name remains URL-derived. A working level keeps its own cache namespace
    // (its chunk levels are renumbered), but not its own volume ID.
    vol->remoteCacheId_ = vol->id();
    if (spec.baseScaleLevel > 0) {
        vol->remoteCacheId_ = deriveRebasedRemoteCacheId(
            vol->remoteCacheId_, remoteUrl, spec.baseScaleLevel);
    }

    if (metadata.is_object()) {
        if (metadata.contains("voxelsize") && metadata["voxelsize"].is_number()) {
            const double value = metadata["voxelsize"].get_double();
            vol->hasExplicitVoxelSizeOverride_ = std::isfinite(value) && value > 0.0;
        }
        vol->metadata_.update(metadata);
        vol->metadata_["width"] = vol->_width;
        vol->metadata_["height"] = vol->_height;
        vol->metadata_["slices"] = vol->_slices;
    }

    if (spec.baseScaleLevel > 0) {
        std::optional<double> logicalOverride;
        if (metadata.is_object() && metadata.contains("voxelsize") &&
            metadata["voxelsize"].is_number()) {
            const double value = metadata["voxelsize"].get_double();
            if (std::isfinite(value) && value > 0.0)
                logicalOverride = value;
        }
        if (logicalOverride) {
            vol->metadata_["voxelsize"] = *logicalOverride;
        } else if (nativeRemoteVoxelSize) {
            vol->metadata_["voxelsize"] =
                *nativeRemoteVoxelSize * static_cast<double>(std::uint64_t{1} << spec.baseScaleLevel);
        } else {
            throw std::runtime_error(
                "rebased remote volume requires a positive logical voxel-size override "
                "or trustworthy native voxel size metadata");
        }
    }

    return vol;
}

std::filesystem::path Volume::remotePersistentCachePath() const
{
    if (!isRemote_ || remoteCacheRoot_.empty())
        return {};
    return remoteCacheRoot_ / remoteCacheId_;
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
    // 0 means unknown. A document may state the key as null, as text, or not
    // at all, and a physical measurement must degrade to "unavailable" rather
    // than throw from deep inside whatever asked for it.
    const double value = vc::json::number_or(metadata_, "voxelsize", 0.0);
    return std::isfinite(value) && value > 0.0 ? value : 0.0;
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
    return sharedChunkCache().get();
}

std::shared_ptr<vc::render::ChunkCache> Volume::sharedChunkCache()
{
    std::lock_guard<std::mutex> lock(cacheMutex_);
    if (!chunkedCache_) {
        vc::render::ChunkCache::Options options;
        chunkedCache_ = createChunkCacheConfigured(std::move(options));
        if (!chunkedCache_) {
            throw std::runtime_error("Volume::chunkedCache failed to create chunk cache");
        }
    }
    return chunkedCache_;
}

std::shared_ptr<vc::render::ChunkCache> Volume::createChunkCacheConfigured(
    vc::render::ChunkCache::Options options) const
{
    if (isRemote_ && !remoteCacheRoot_.empty())
        options.persistentCachePath = remotePersistentCachePath();
    if (options.persistentCachePath &&
        vc::render::PersistentZarrCacheBudget::findForPath(*options.persistentCachePath)) {
        options.persistentCacheBudgetRoot = remoteCacheRoot_;
    }

    vc::render::OpenedChunkedZarr opened;
    if (isRemote_) {
        opened = baseScaleLevel_ > 0
            ? vc::render::openHttpZarrPyramid(
                  remoteUrl_, remoteAuth_, baseScaleLevel_)
            : vc::render::openHttpZarrPyramid(remoteUrl_, remoteAuth_);
    } else {
        opened = vc::render::openLocalZarrPyramid(path_);
    }

    if (opened.fetchers.empty()) {
        return nullptr;
    }

    options.zarrMirrorMetadata = std::move(opened.zarrMirrorMetadata);
    auto levels = makeChunkCacheLevelInfo(opened);
    return vc::render::processChunkCacheService()->acquireSource(
        chunkCacheSourceIdentity(), std::move(levels),
        std::move(opened.fetchers), opened.fillValue, opened.dtype,
        std::move(options));
}

std::string Volume::chunkCacheSourceIdentity() const
{
    if (isRemote_) {
        const auto normalized = vc::core::util::remoteFileCacheSource(
            vc::core::util::normalizeRemoteFileLocation(remoteUrl_));
        std::string identity = "remote|" + normalized +
            "|base=" + std::to_string(baseScaleLevel_);
        auto persistentPath = remotePersistentCachePath();
        if (!persistentPath.empty()) {
            std::error_code ec;
            const auto absolute = std::filesystem::absolute(persistentPath, ec);
            if (!ec)
                persistentPath = absolute;
            identity += "|cache=" +
                persistentPath.lexically_normal().generic_string();
        }
        return identity;
    }
    std::error_code ec;
    auto sourcePath = std::filesystem::weakly_canonical(path_, ec);
    if (ec)
        sourcePath = std::filesystem::absolute(path_, ec);
    if (ec)
        sourcePath = path_;
    return "local|" + sourcePath.lexically_normal().string() +
           "|base=" + std::to_string(baseScaleLevel_);
}

void Volume::invalidateCache()
{
    std::lock_guard<std::mutex> lock(cacheMutex_);
    if (chunkedCache_) {
        chunkedCache_->invalidate();
    } else {
        vc::render::processChunkCacheService()->invalidateSource(
            chunkCacheSourceIdentity());
    }
    chunkedCache_.reset();
}

// ============================================================================
// Sampling API
// ============================================================================

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

std::shared_ptr<utils::ZarrArray>
Volume::cachedZarrArrayForRead(int level) const
{
    std::lock_guard<std::mutex> lk(readArrayCacheMutex_);
    if (readArrayCache_.size() <= static_cast<size_t>(level)) {
        readArrayCache_.resize(static_cast<size_t>(level) + 1);
    }
    auto& slot = readArrayCache_[static_cast<size_t>(level)];
    if (!slot) {
        slot = std::make_shared<utils::ZarrArray>(
            openLocalZarrArrayForRead(zarrArrayPathForLevel(path(), level),
                                      static_cast<int>(dtypeSize())));
    }
    return slot;
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

    auto arrayPtr = cachedZarrArrayForRead(level);
    return arrayPtr->read_chunk(chunkZYX);
}

bool Volume::readChunkInto(
    int level,
    const std::array<size_t, 3>& chunkZYX,
    std::span<std::byte> output) const
{
    if (isRemote())
        throw std::runtime_error("Volume::readChunkInto is only supported for local zarr volumes");
    if (level < 0)
        throw std::out_of_range("Volume::readChunkInto level must be non-negative");
    if (!hasScaleLevel(level))
        throw std::out_of_range("Volume::readChunkInto requested missing zarr scale level " + std::to_string(level));

    auto arrayPtr = cachedZarrArrayForRead(level);
    return arrayPtr->read_chunk_into(chunkZYX, output);
}

size_t Volume::chunkByteSize(int level) const
{
    const auto cs = chunkShape(level);
    return static_cast<size_t>(cs[0]) * static_cast<size_t>(cs[1]) *
           static_cast<size_t>(cs[2]) * dtypeSize();
}

std::vector<std::byte> Volume::readChunkOrFill(
    int level,
    const std::array<size_t, 3>& chunkZYX) const
{
    if (auto chunk = readChunk(level, chunkZYX))
        return std::move(*chunk);

    auto arrayPtr = cachedZarrArrayForRead(level);
    return filledChunkBytes(*arrayPtr);
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

    auto arrayPtr = cachedZarrArrayForRead(level);
    if (arrayPtr->is_sharded())
        return arrayPtr->inner_chunk_exists(chunkZYX);
    return arrayPtr->chunk_exists(chunkZYX);
}

