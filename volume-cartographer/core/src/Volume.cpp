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
#include "vc_reader/zarr.hpp"

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
    vol->remoteSourceCacheId_ = vol->remoteCacheId_;
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

std::string Volume::sourceLocation() const
{
    if (!isRemote_)
        return path_.string();
    if (remoteAuth_.empty())
        return remoteUrl_;
    // Signed requests name the bucket: https://BUCKET.s3[.REGION].amazonaws.com/KEY.
    const auto scheme = remoteUrl_.find("://");
    const std::string rest = scheme == std::string::npos ? remoteUrl_ : remoteUrl_.substr(scheme + 3);
    const auto slash = rest.find('/');
    const std::string host = rest.substr(0, slash);
    const auto s3 = host.find(".s3.");
    if (s3 == std::string::npos || host.find(".amazonaws.com") == std::string::npos)
        throw std::runtime_error("signed requests need an S3 virtual-host URL: " + remoteUrl_);
    return "s3://" + host.substr(0, s3) + (slash == std::string::npos ? std::string() : rest.substr(slash));
}

vcr::zarr::OpenOptions Volume::sourceOpenOptions() const
{
    vcr::zarr::OpenOptions options;
    if (!isRemote_)
        return options;
    if (!remoteAuth_.empty())
        options.access = vcr::zarr::s3Access(sourceLocation(),
            vcr::Auth{"s3", remoteAuth_.access_key, remoteAuth_.secret_key,
                      remoteAuth_.session_token, remoteAuth_.region});
    if (!remoteCacheRoot_.empty())
        options.vc3dVolumeDir = remoteCacheRoot_ / remoteSourceCacheId_;
    return options;
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

