#include "vc/core/types/Volume.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
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
