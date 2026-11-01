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
#include "vc/core/util/RemoteUrl.hpp"
#include "vc/core/util/RemoteCacheSettings.hpp"
#include "utils/hash.hpp"
#include "vc_reader/zarr.hpp"
#include "vc_reader/zarr_metadata.hpp"

static const std::filesystem::path METADATA_FILE = "meta.json";
static const std::filesystem::path METADATA_FILE_ALT = "metadata.json";

namespace
{

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

// The first sidecar document a remote store has, with `scan` flattened into its root.
std::optional<utils::Json> loadRemoteVolumeMetadata(const vcr::zarr::StoreMetadata& store, const std::string& location)
{
    for (const auto& name : {METADATA_FILE.string(), METADATA_FILE_ALT.string()}) {
        const auto document = store.documents.find(name);
        if (document == store.documents.end() || !document->is_object())
            continue;
        auto json = utils::Json::parse(document->dump());
        if (json.contains("scan") && json["scan"].is_object())
            json.update(json["scan"]);
        if (!json.contains("format"))
            json["format"] = "zarr";
        applyResolvedVoxelSize(json, store.voxelSizeUm, location + "/" + name);
        return json;
    }
    return std::nullopt;
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

int ceilDivPow2(int v, int level)
{
    const int64_t denom = int64_t{1} << level;
    return static_cast<int>((static_cast<int64_t>(v) + denom - 1) / denom);
}

bool paddedShapeOK(long long actual, long long expected, int padMultiple)
{
    const long long maxPad = std::max(1, padMultiple);
    return actual >= expected && actual - expected < maxPad;
}

// The signed form of an S3 URL.
std::string signedS3Location(const std::string& url)
{
    auto location = vc::s3Location(url);
    if (!location)
        throw std::runtime_error("signed requests need an S3 virtual-host URL: " + url);
    return std::move(*location);
}

std::array<int, 3> shape3(const vcr::Index& shape, const std::string& description)
{
    if (shape.size() != 3)
        throw std::runtime_error("zarr array must be 3D: " + description);
    return {static_cast<int>(shape[0]), static_cast<int>(shape[1]), static_cast<int>(shape[2])};
}

// One level's shape per level number, zero for absent numbers.
std::vector<std::array<int, 3>> levelShapes(const std::vector<vcr::zarr::PyramidLevelInfo>& levels,
                                            const std::string& description)
{
    std::vector<std::array<int, 3>> shapes;
    for (const auto& level : levels) {
        if (level.meta.dtype != levels.front().meta.dtype)
            throw std::runtime_error("levels of " + description + " have different dtypes");
        if (shapes.size() <= static_cast<std::size_t>(level.level))
            shapes.resize(static_cast<std::size_t>(level.level) + 1, {0, 0, 0});
        shapes[static_cast<std::size_t>(level.level)] = shape3(level.meta.shape, description);
    }
    return shapes;
}

// A working level is the coarse end of a gapless dyadic pyramid with one fill value.
void validateWorkingLevel(const std::vector<vcr::zarr::PyramidLevelInfo>& levels, int base, const std::string& description)
{
    for (std::size_t index = 0; index < levels.size(); ++index) {
        if (levels[index].level != static_cast<int>(index))
            throw std::runtime_error("VC pyramid numeric groups contain a gap at /" + std::to_string(index) +
                                     " of " + description);
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const auto first = static_cast<int>(levels.front().meta.shape[axis]);
            if (!paddedShapeOK(static_cast<long long>(levels[index].meta.shape[axis]),
                               ceilDivPow2(first, static_cast<int>(index)),
                               static_cast<int>(levels[index].meta.chunks[axis])))
                throw std::runtime_error("VC pyramid group /" + std::to_string(index) +
                                         " shape is not a dyadic downscale of /0 within storage-chunk padding tolerance");
        }
    }
    if (base >= static_cast<int>(levels.size()))
        throw std::runtime_error("requested VC pyramid base group /" + std::to_string(base) + " is unavailable");
    for (std::size_t index = static_cast<std::size_t>(base); index < levels.size(); ++index)
        if (levels[index].meta.fill != levels[static_cast<std::size_t>(base)].meta.fill)
            throw std::runtime_error("VC pyramid retained groups must have one consistent fill value");
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
        const auto voxelSize = vcr::zarr::voxelSizeUm(vcr::zarr::Json::parse(full.dump()));
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

void Volume::zarrOpen()
{
    if (!metadata_.contains("format") || metadata_["format"].get_string() != "zarr")
        return;

    const auto levels = vcr::zarr::pyramidLevels(path_.string(), {});
    if (levels.empty()) {
        throw std::runtime_error("no physical zarr dataset directories found in " + path_.string());
    }
    const auto levelShapesZYX = levelShapes(levels, path_.string());
    // The padding a level's shape may have over its downscaled dimensions.
    std::vector<std::array<int, 3>> storageChunkShapes(levelShapesZYX.size(), {128, 128, 128});
    for (const auto& level : levels)
        storageChunkShapes[static_cast<std::size_t>(level.level)] = shape3(level.meta.chunks, path_.string());

    if (metadataAutoGenerated_) {
        bool hasReference = false;
        int baseSlices = 0;
        int baseHeight = 0;
        int baseWidth = 0;

        for (size_t level = 0; level < levelShapesZYX.size(); ++level) {
            const auto& shape = levelShapesZYX[level];
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

            const auto& storageChunkShape = storageChunkShapes[level];
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
        for (size_t level = 0; level < levelShapesZYX.size(); ++level) {
            const auto& shape = levelShapesZYX[level];
            if (shape[0] == 0 && shape[1] == 0 && shape[2] == 0) {
                continue;
            }
            hasAnyPhysicalScale = true;

            const int expectedSlices = ceilDivPow2(_slices, static_cast<int>(level));
            const int expectedHeight = ceilDivPow2(_height, static_cast<int>(level));
            const int expectedWidth = ceilDivPow2(_width, static_cast<int>(level));

            const auto& storageChunkShape = storageChunkShapes[level];
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
    const vcr::Auth& authIn,
    const utils::Json& metadata,
    bool discoverAwsCredentials)
{
    const auto spec = vc::parseRemoteVolumeSpec(url);
    const std::string& remoteUrl = spec.sourceUrl;
    auto auth = authIn;

    // Recognized S3 sources are tried anonymously, then with the given or
    // discovered credentials only when anonymous access fails.
    const auto openLevels = [&](const vcr::Auth& requestAuth) {
        vcr::zarr::OpenOptions options;
        std::string location = remoteUrl;
        if (!requestAuth.kind.empty()) {
            location = signedS3Location(remoteUrl);
            options.access = vcr::zarr::s3Access(location, requestAuth);
        }
        options.cacheDirs = {vc::settings::remoteCachePath().string()};
        return std::pair{vcr::zarr::pyramidLevels(location, options), options};
    };
    decltype(openLevels(auth)) opened;
    if (!spec.useAwsSigv4) {
        opened = openLevels(auth);
    } else {
        try {
            opened = openLevels({});
            auth = {};
        } catch (const std::exception&) {
            if (auth.kind.empty() && discoverAwsCredentials) {
                auth = vcr::zarr::awsCredentials().value_or(vcr::Auth{});
            }
            if (auth.kind.empty())
                throw;
            opened = openLevels(auth);
        }
    }
    auto& [allLevels, openOptions] = opened;
    if (allLevels.empty())
        throw std::runtime_error("No zarr levels found at " + remoteUrl);
    if (spec.baseScaleLevel > 0)
        validateWorkingLevel(allLevels, spec.baseScaleLevel, remoteUrl);
    auto shapes = levelShapes(allLevels, remoteUrl);
    // A working level's voxels are its level 0: drop the finer physical levels.
    shapes.erase(shapes.begin(), shapes.begin() + spec.baseScaleLevel);

    int firstPresentLevel = -1;
    for (std::size_t level = 0; level < shapes.size(); ++level) {
        const auto& shape = shapes[level];
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
    const auto& firstShape = shapes[static_cast<std::size_t>(firstPresentLevel)];
    const size_t firstScale = spec.baseScaleLevel > 0 ? 1 : size_t{1} << firstPresentLevel;
    vol->_slices = static_cast<int>(static_cast<size_t>(firstShape[0]) * firstScale);
    vol->_height = static_cast<int>(static_cast<size_t>(firstShape[1]) * firstScale);
    vol->_width = static_cast<int>(static_cast<size_t>(firstShape[2]) * firstScale);

    const auto id = vcr::zarr::vc3dVolumeId(remoteUrl);
    vol->metadata_["uuid"] = id;
    vol->metadata_["name"] = id.substr(0, id.rfind('-'));
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
        const auto store = vcr::zarr::volumeMetadata(vol->sourceLocation(), openOptions);
        if (auto remoteMeta = loadRemoteVolumeMetadata(store, vol->sourceLocation())) {
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
    // name remains URL-derived. Every working level shares the physical pyramid's
    // cache.
    vol->remoteSourceCacheId_ = vol->id();

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
    return remoteAuth_.kind.empty() ? remoteUrl_ : signedS3Location(remoteUrl_);
}

vcr::zarr::OpenOptions Volume::sourceOpenOptions() const
{
    if (!isRemote_)
        return {};
    // Chunks live at <root>/<scheme>/<bucket>/<path>, shared with every vc_reader
    // user of the root. A volume cached by an earlier VC3D keeps its own folder.
    auto options = vc::settings::remoteOpenOptions(remoteCacheRoot_);
    if (!remoteAuth_.kind.empty())
        options.access = vcr::zarr::s3Access(sourceLocation(), remoteAuth_);
    if (!remoteCacheRoot_.empty()) {
        options.compressCache = vc::settings::remoteCacheDelta3dEnabled();
        if (const auto folder = remoteCacheRoot_ / remoteSourceCacheId_; std::filesystem::is_directory(folder))
            options.vc3dVolumeDir = folder;
    }
    return options;
}

std::filesystem::path Volume::remotePersistentCachePath() const
{
    if (!isRemote_ || remoteCacheRoot_.empty())
        return {};
    const auto folder = remoteCacheRoot_ / remoteSourceCacheId_;
    return std::filesystem::is_directory(folder) ? folder : vcr::zarr::decodedCachePath(remoteCacheRoot_, sourceLocation());
}

int Volume::sliceWidth() const noexcept { return _width; }
int Volume::sliceHeight() const noexcept { return _height; }
int Volume::numSlices() const noexcept { return _slices; }
std::array<int, 3> Volume::shape() const noexcept { return {_slices, _height, _width}; }
std::array<int, 3> Volume::shapeXyz() const noexcept { return {_width, _height, _slices}; }
double Volume::voxelSize() const
{
    // 0 means unknown. A document may state the key as null, as text, or not
    // at all, and a physical measurement must degrade to "unavailable" rather
    // than throw from deep inside whatever asked for it.
    const double value = vc::json::number_or(metadata_, "voxelsize", 0.0);
    return std::isfinite(value) && value > 0.0 ? value : 0.0;
}
