#include "vc/core/util/QuadSurface.hpp"

#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/LoadJson.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/core/util/Tiff.hpp"
#include <vc_object_store/Tiff.hpp>
#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/geometry/Geometry.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <system_error>
#include <cmath>
#include <limits>
#include <cerrno>
#include <algorithm>
#include <thread>
#include <vector>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <exception>
#include <set>
#include <unordered_map>

#include <stdio.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#ifdef __linux__
// renameat2(RENAME_EXCHANGE) in save() needs the GNU prototypes.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <fcntl.h>
#include <unistd.h>
#endif

#include "vc/core/util/MemMap.hpp"

// Use libtiff for BigTIFF
#include <tiffio.h>

// GCC/Clang spell the restrict qualifier __restrict__; MSVC only has __restrict.
#if defined(_MSC_VER)
#define __restrict__ __restrict
#endif

namespace {

void replaceFile(const std::filesystem::path& source,
                 const std::filesystem::path& destination)
{
#ifdef _WIN32
    if (!MoveFileExW(source.wstring().c_str(),
                     destination.wstring().c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const std::error_code ec(static_cast<int>(GetLastError()),
                                 std::system_category());
        throw std::filesystem::filesystem_error(
            "cannot replace file", source, destination, ec);
    }
#else
    std::filesystem::rename(source, destination);
#endif
}

constexpr auto kSaveRollbackDir = ".vc-save-rollback";
constexpr auto kSaveRollbackReady = ".ready";

std::recursive_mutex& surfaceDirWriteMutex(const std::filesystem::path& dir)
{
    // Keyed by normalized dir string so separate QuadSurface objects pointing
    // at the same segment dir share one lock. Recursive because saveOverwrite
    // holds it across nested saveSnapshot()/save() calls, and Windows
    // publication performs recovery while the save lock is already held.
    static std::mutex mapMutex;
    static std::unordered_map<std::string, std::recursive_mutex> locks;
    const std::string key = dir.lexically_normal().string();
    std::lock_guard<std::mutex> guard(mapMutex);
    return locks[key];
}

void recoverIncompleteSave(const std::filesystem::path& destination)
{
    // Loading and publication must be mutually exclusive. In particular, a
    // loader must not mistake an active Windows rollback journal for one left
    // by an interrupted process and restore it while files are being replaced.
    std::lock_guard<std::recursive_mutex> dirLock(
        surfaceDirWriteMutex(destination));

    const auto rollback = destination / kSaveRollbackDir;
    if (!std::filesystem::exists(rollback)) {
        return;
    }

    const auto ready = rollback / kSaveRollbackReady;
    if (!std::filesystem::exists(ready)) {
        // Publication never began, or it committed and only cleanup was
        // interrupted. In either case the destination is already coherent.
        std::filesystem::remove_all(rollback);
        return;
    }

    std::vector<std::filesystem::path> backups;
    std::set<std::filesystem::path> originalNames;
    for (const auto& entry : std::filesystem::directory_iterator(rollback)) {
        if (entry.is_regular_file() && entry.path().filename() != kSaveRollbackReady) {
            backups.push_back(entry.path());
            originalNames.insert(entry.path().filename());
        }
    }

    // Remove files introduced by the interrupted generation before restoring
    // every original. Keep the backups intact until recovery is complete so a
    // second interruption can safely retry the same recovery.
    for (const auto& entry : std::filesystem::directory_iterator(destination)) {
        if (entry.is_regular_file() && !originalNames.contains(entry.path().filename())) {
            std::filesystem::remove(entry.path());
        }
    }

    const auto restoreDir = rollback / ".restore";
    std::filesystem::create_directories(restoreDir);
    for (const auto& backup : backups) {
        const auto staged = restoreDir / backup.filename();
        std::filesystem::copy_file(
            backup, staged, std::filesystem::copy_options::overwrite_existing);
        replaceFile(staged, destination / backup.filename());
    }
    std::filesystem::remove_all(rollback);
}

#ifdef _WIN32
void createRollbackBackup(const std::filesystem::path& original,
                          const std::filesystem::path& backup)
{
    // A same-volume hard link preserves the old file object when the live name
    // is replaced, without copying multi-gigabyte TIFFs on every autosave.
    // Fall back to a copy on filesystems that do not support hard links.
    if (::CreateHardLinkW(backup.wstring().c_str(), original.wstring().c_str(), nullptr)) {
        return;
    }
    std::filesystem::copy_file(
        original, backup, std::filesystem::copy_options::overwrite_existing);
}

void replaceDirectoryContents(const std::filesystem::path& source,
                              const std::filesystem::path& destination)
{
    std::filesystem::create_directories(destination);

    // If a previous process stopped during publication, restore its complete
    // prior generation before beginning another save.
    recoverIncompleteSave(destination);

    const auto rollback = destination / kSaveRollbackDir;
    std::filesystem::create_directories(rollback);
    for (const auto& entry : std::filesystem::directory_iterator(destination)) {
        if (entry.is_regular_file()) {
            createRollbackBackup(entry.path(), rollback / entry.path().filename());
        }
    }

    // This marker is created only after every original file is durable in the
    // rollback directory. Its presence means publication may have begun.
    const auto ready = rollback / kSaveRollbackReady;
    {
        std::ofstream marker(ready, std::ios::binary);
        marker << "ready\n";
        if (!marker) {
            throw std::runtime_error("failed to prepare save rollback journal: " +
                                     ready.string());
        }
    }

    // Windows cannot reliably remove a directory while files from it are
    // mapped by another QuadSurface. The mappings opt into FILE_SHARE_DELETE,
    // so replace the completed files individually instead. The temporary
    // directory already contains newly written x/y/z/meta files plus copies of
    // every regular auxiliary file from the destination.
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(source)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path());
        }
    }
    // Do not mutate a directory while iterating it: directory_iterator may
    // otherwise skip entries on Windows. Publish metadata last so readers do
    // not observe it before the corresponding data files are replaced.
    std::sort(files.begin(), files.end(), [](const auto& lhs, const auto& rhs) {
        const bool lhsMeta = lhs.filename() == "meta.json";
        const bool rhsMeta = rhs.filename() == "meta.json";
        return lhsMeta != rhsMeta ? !lhsMeta : lhs.filename() < rhs.filename();
    });
    try {
        for (const auto& file : files) {
            replaceFile(file, destination / file.filename());
        }

        // Removing the marker commits the complete new generation. If cleanup
        // is interrupted after this point, recovery keeps the new files.
        std::filesystem::remove(ready);
    } catch (...) {
        const auto error = std::current_exception();
        try {
            recoverIncompleteSave(destination);
        } catch (const std::exception& recoveryError) {
            Logger()->error("failed to roll back interrupted save for {}: {}",
                            destination.string(), recoveryError.what());
        }
        std::rethrow_exception(error);
    }

    std::error_code cleanupEc;
    std::filesystem::remove_all(rollback, cleanupEc);
    if (cleanupEc) {
        Logger()->warn("could not remove save rollback directory {}: {}",
                       rollback.string(), cleanupEc.message());
    }
    cleanupEc.clear();
    std::filesystem::remove_all(source, cleanupEc);
    if (cleanupEc) {
        Logger()->warn("could not remove completed save temp directory {}: {}",
                       source.string(), cleanupEc.message());
    }
}
#endif

// Channel files with a sample count OpenCV's TIFF codec does not handle.
bool readSamplesTiffFirst(const std::filesystem::path& path)
{
    std::unique_ptr<TIFF, decltype(&TIFFClose)> tif(TIFFOpen(path.string().c_str(), "r"), TIFFClose);
    uint16_t samples = 1;
    if (tif) TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &samples);
    return samples == 2 || samples > 4;
}

// OpenCV depths as TIFF samples (TiffSamples::bits, sampleFormat).
bool tiffSampleFormat(int depth, uint16_t& bits, uint16_t& format)
{
    switch (depth) {
    case CV_8U: bits = 8; format = SAMPLEFORMAT_UINT; return true;
    case CV_16U: bits = 16; format = SAMPLEFORMAT_UINT; return true;
    case CV_32S: bits = 32; format = SAMPLEFORMAT_INT; return true;
    case CV_32F: bits = 32; format = SAMPLEFORMAT_IEEEFP; return true;
    case CV_64F: bits = 64; format = SAMPLEFORMAT_IEEEFP; return true;
    }
    return false;
}

// A channel image with any number of components through the store's TIFF
// sample codec; empty when its samples have no OpenCV depth.
cv::Mat readChannelSamples(const std::filesystem::path& path)
{
    const auto samples = readTiffSamples(path, false);
    for (const int depth : {CV_8U, CV_16U, CV_32S, CV_32F, CV_64F}) {
        uint16_t bits = 0, format = 0;
        if (!tiffSampleFormat(depth, bits, format) || bits != samples.bits || format != samples.sampleFormat ||
            samples.components > CV_CN_MAX)
            continue;
        cv::Mat image(int(samples.height), int(samples.width), CV_MAKETYPE(depth, samples.components));
        std::memcpy(image.data, samples.bytes.data(), samples.bytes.size());
        return image;
    }
    return {};
}

void writeChannelSamples(const std::filesystem::path& path, const cv::Mat& image)
{
    TiffSamples samples;
    if (image.empty() || !tiffSampleFormat(image.depth(), samples.bits, samples.sampleFormat))
        throw std::runtime_error("unsupported image for " + path.string());
    const cv::Mat contiguous = image.isContinuous() ? image : image.clone();
    samples.width = uint32_t(contiguous.cols);
    samples.height = uint32_t(contiguous.rows);
    samples.components = uint16_t(contiguous.channels());
    samples.bytes.assign(contiguous.datastart, contiguous.dataend);
    writeTiffSamples(path, samples);
}

void normalizeMaskChannel(cv::Mat& mask)
{
    if (mask.empty()) {
        return;
    }

    cv::Mat singleChannel;
    if (mask.channels() == 1) {
        singleChannel = mask;
    } else if (mask.channels() == 3) {
        cv::cvtColor(mask, singleChannel, cv::COLOR_BGR2GRAY);
    } else if (mask.channels() == 4) {
        cv::cvtColor(mask, singleChannel, cv::COLOR_BGRA2GRAY);
    } else {
        std::vector<cv::Mat> channels;
        cv::split(mask, channels);
        if (!channels.empty()) {
            singleChannel = channels[0];
        } else {
            singleChannel = mask;
        }
    }

    if (singleChannel.depth() != CV_8U) {
        cv::Mat converted;
        singleChannel.convertTo(converted, CV_8U);
        singleChannel = converted;
    }

    mask = singleChannel;
}

inline bool isValidPointSample(const cv::Vec3f& point)
{
    return point[0] != -1.0f && point[1] != -1.0f && point[2] != -1.0f
        && std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]);
}

float scaledCenterComponent(int count, float scale)
{
    if (!std::isfinite(scale) || scale == 0.0f) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return static_cast<float>(static_cast<double>(count) / 2.0 / static_cast<double>(scale));
}

cv::Vec3f surfaceCenterFor(const cv::Mat_<cv::Vec3f>& points, const cv::Vec2f& scale)
{
    return {
        scaledCenterComponent(points.cols, scale[0]),
        scaledCenterComponent(points.rows, scale[1]),
        0.f
    };
}

struct LazySurfaceGeometry {
    cv::Vec2f scale;
    cv::Vec3f center;
    cv::Rect bounds;
};

LazySurfaceGeometry readLazySurfaceGeometry(const std::filesystem::path& path,
                                            const utils::Json& metadata)
{
    auto readScale = [](const utils::Json& json) -> std::optional<cv::Vec2f> {
        if (!json.contains("scale") || !json["scale"].is_array() ||
            json["scale"].size() < 2) {
            return std::nullopt;
        }
        const cv::Vec2f scale{json["scale"][0].get_float(),
                              json["scale"][1].get_float()};
        if (!std::isfinite(scale[0]) || !std::isfinite(scale[1]) ||
            scale[0] <= 0.0f || scale[1] <= 0.0f) {
            return std::nullopt;
        }
        return scale;
    };

    // The caller's metadata may be a partial override (second constructor), so
    // fall back to the on-disk meta.json for missing fields lazily if needed.
    std::optional<utils::Json> diskMetadataStorage;
    auto diskMetadata = [&]() -> const utils::Json& {
        if (!diskMetadataStorage) {
            diskMetadataStorage = vc::json::load_json_file(path / "meta.json");
        }
        return *diskMetadataStorage;
    };

    std::optional<cv::Vec2f> scale = readScale(metadata);
    if (!scale) {
        scale = readScale(diskMetadata());
    }
    if (!scale) {
        throw std::runtime_error("Missing or invalid surface scale in: " +
                                 (path / "meta.json").string());
    }

    uint32_t width = 0;
    uint32_t height = 0;
    const auto xPath = path / "x.tif";
    TIFF* tif = TIFFOpen(xPath.string().c_str(), "r");
    if (!tif) {
        throw std::runtime_error("Failed to open TIFF: " + xPath.string());
    }
    const bool haveGeometry =
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width) &&
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    TIFFClose(tif);
    if (!haveGeometry || width == 0 || height == 0) {
        throw std::runtime_error("TIFF missing width/height: " + xPath.string());
    }

    LazySurfaceGeometry result;
    result.scale = *scale;
    result.center = {
        scaledCenterComponent(static_cast<int>(width), result.scale[0]),
        scaledCenterComponent(static_cast<int>(height), result.scale[1]),
        0.0f,
    };
    result.bounds = {
        0, 0, static_cast<int>(width) - 1, static_cast<int>(height) - 1,
    };
    return result;
}

cv::Mat_<cv::Vec3f> warpAffinePointsLinearPreservingInvalids(
    const cv::Mat_<cv::Vec3f>& points,
    const cv::Mat& srcToDst,
    const cv::Size& dstSize)
{
    cv::Mat_<cv::Vec3f> warped(dstSize.height, dstSize.width,
                               cv::Vec3f(-1.0f, -1.0f, -1.0f));
    if (points.empty() || dstSize.width <= 0 || dstSize.height <= 0) {
        return warped;
    }

    cv::Mat dstToSrc;
    cv::invertAffineTransform(srcToDst, dstToSrc);

    // Hoist the six affine coefficients out of the per-pixel loop. The Mat is
    // loop-invariant, but the compiler cannot prove dstToSrc.at<double>() does
    // not alias the warped store, so it reloads all six entries every pixel.
    // Reading them into locals is bit-identical and lets each row run
    // independently under OpenMP. (The mapping is NOT x-separable here — srcX
    // depends on both dstX and dstY — so only the coefficient hoist applies,
    // and the explicit m*dstX form is kept to preserve float rounding.)
    const double m00 = dstToSrc.at<double>(0, 0);
    const double m01 = dstToSrc.at<double>(0, 1);
    const double m02 = dstToSrc.at<double>(0, 2);
    const double m10 = dstToSrc.at<double>(1, 0);
    const double m11 = dstToSrc.at<double>(1, 1);
    const double m12 = dstToSrc.at<double>(1, 2);

    constexpr double kBoundsEpsilon = 1e-5;
    #pragma omp parallel for schedule(static)
    for (int dstY = 0; dstY < dstSize.height; ++dstY) {
        cv::Vec3f* dstRow = warped[dstY];
        for (int dstX = 0; dstX < dstSize.width; ++dstX) {
            double srcX = m00 * dstX + m01 * dstY + m02;
            double srcY = m10 * dstX + m11 * dstY + m12;

            if (srcX < -kBoundsEpsilon || srcY < -kBoundsEpsilon
                || srcX > static_cast<double>(points.cols - 1) + kBoundsEpsilon
                || srcY > static_cast<double>(points.rows - 1) + kBoundsEpsilon) {
                continue;
            }

            srcX = std::clamp(srcX, 0.0, static_cast<double>(points.cols - 1));
            srcY = std::clamp(srcY, 0.0, static_cast<double>(points.rows - 1));
            const int x0 = static_cast<int>(std::floor(srcX));
            const int y0 = static_cast<int>(std::floor(srcY));
            const int x1 = std::min(x0 + 1, points.cols - 1);
            const int y1 = std::min(y0 + 1, points.rows - 1);
            const float fx = static_cast<float>(srcX - static_cast<double>(x0));
            const float fy = static_cast<float>(srcY - static_cast<double>(y0));

            const cv::Vec3f& p00 = points(y0, x0);
            const cv::Vec3f& p01 = points(y0, x1);
            const cv::Vec3f& p10 = points(y1, x0);
            const cv::Vec3f& p11 = points(y1, x1);
            if (!isValidPointSample(p00) || !isValidPointSample(p01)
                || !isValidPointSample(p10) || !isValidPointSample(p11)) {
                continue;
            }

            const cv::Vec3f top = p00 * (1.0f - fx) + p01 * fx;
            const cv::Vec3f bottom = p10 * (1.0f - fx) + p11 * fx;
            dstRow[dstX] = top * (1.0f - fy) + bottom * fy;
        }
    }

    return warped;
}

} // namespace

static cv::Vec3f internal_loc(const cv::Vec3f &nominal, const cv::Vec3f &internal, const cv::Vec2f &scale)
{
    return internal + cv::Vec3f(nominal[0]*scale[0], nominal[1]*scale[1], nominal[2]);
}

static cv::Vec3f nominal_loc(const cv::Vec3f &nominal, const cv::Vec3f &internal, const cv::Vec2f &scale)
{
    return nominal + cv::Vec3f(internal[0]/scale[0], internal[1]/scale[1], internal[2]);
}


QuadSurface::QuadSurface(const cv::Mat_<cv::Vec3f> &points, const cv::Vec2f &scale)
{
    _points = std::make_unique<cv::Mat_<cv::Vec3f>>(points.clone());
    _bounds = {0,0,_points->cols-1,_points->rows-1};
    _scale = scale;
    _center = surfaceCenterFor(*_points, _scale);
}

QuadSurface::QuadSurface(cv::Mat_<cv::Vec3f> *points, const cv::Vec2f &scale)
{
    _points.reset(points);
    //-1 as many times we read with linear interpolation and access +1 locations
    _bounds = {0,0,_points->cols-1,_points->rows-1};
    _scale = scale;
    _center = surfaceCenterFor(*_points, _scale);
}

namespace {
static Rect3D rect_from_json(const utils::Json &json)
{
    return {{json[0][0].get_float(),json[0][1].get_float(),json[0][2].get_float()},
            {json[1][0].get_float(),json[1][1].get_float(),json[1][2].get_float()}};
}
} // anonymous namespace

std::vector<std::string> tifxyz_tags(const utils::Json& meta)
{
    std::vector<std::string> tags;
    if (!meta.is_object() || !meta.contains("tags") || !meta["tags"].is_object()) return tags;
    const auto& recorded = meta["tags"];
    for (auto it = recorded.begin(); it != recorded.end(); ++it) tags.push_back(it.key());
    return tags;
}

std::vector<std::pair<int, int>> surface_components(const utils::Json& meta)
{
    std::vector<std::pair<int, int>> components;
    if (meta.contains("components") && meta["components"].is_array()) {
        for (const auto& c : meta["components"]) {
            if (c.is_array() && c.size() >= 2)
                components.emplace_back(c[0].get_int(), c[1].get_int());
        }
    }
    return components;
}

QuadSurface::QuadSurface(const std::filesystem::path &path_)
{
    recoverIncompleteSave(path_);
    path = path_;
    id = path_.filename().string();
    auto metaPath = path_ / "meta.json";
    meta = vc::json::load_json_file(metaPath);

    if (meta.contains("bbox"))
        _bbox = rect_from_json(meta["bbox"]);

    _components = surface_components(meta);
    tags = tifxyz_tags(meta);

    const LazySurfaceGeometry geometry = readLazySurfaceGeometry(path, meta);
    _scale = geometry.scale;
    _center = geometry.center;
    _bounds = geometry.bounds;
    _needsLoad = true;  // Points will be loaded lazily
}

QuadSurface::QuadSurface(const std::filesystem::path &path_, const utils::Json &json)
{
    recoverIncompleteSave(path_);
    path = path_;
    id = path_.filename().string();
    meta = json;

    if (json.contains("bbox"))
        _bbox = rect_from_json(json["bbox"]);

    _components = surface_components(json);
    tags = tifxyz_tags(json);

    const LazySurfaceGeometry geometry = readLazySurfaceGeometry(path, meta);
    _scale = geometry.scale;
    _center = geometry.center;
    _bounds = geometry.bounds;
    _needsLoad = true;  // Points will be loaded lazily
}

void QuadSurface::ensureLoaded()
{
    // Fast path: already loaded (no lock needed for read)
    if (!_needsLoad) {
        return;
    }

    // Slow path: need to load, acquire lock
    std::lock_guard<std::mutex> lock(_loadMutex);

    // Double-check after acquiring lock (another thread may have loaded)
    if (!_needsLoad) {
        return;
    }

    if (DebugLoggingEnabled()) {
        std::fprintf(stderr, "[SURF] load %s (from %s)\n",
                     id.c_str(), path.string().c_str());
    }
    auto loaded = load_quad_from_tifxyz(path.string());
    if (!loaded) {
        throw std::runtime_error("Failed to load surface from: " + path.string());
    }

    // Transfer ownership of points and other data from loaded surface
    _points = std::move(loaded->_points);

    _bounds = loaded->_bounds;
    _scale = loaded->_scale;
    _center = loaded->_center;
    _channels = std::move(loaded->_channels);

    // Keep existing bbox and meta if already set, otherwise take from loaded
    if (_bbox.low[0] == 0 && _bbox.high[0] == 0) {
        _bbox = loaded->_bbox;
    }


    // Mark as loaded (after all data is set)
    _needsLoad = false;
}

QuadSurface::~QuadSurface() = default;

void QuadSurface::move(cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    ptr += cv::Vec3f(offset[0]*_scale[0], offset[1]*_scale[1], offset[2]);
}

bool QuadSurface::valid(const cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    ensureLoaded();
    cv::Vec3f p = internal_loc(offset+_center, ptr, _scale);
    return loc_valid_xy(*_points, {p[0], p[1]});
}


cv::Vec3f QuadSurface::coord(const cv::Vec3f &ptr, const cv::Vec3f &offset) const
{
    const_cast<QuadSurface*>(this)->ensureLoaded();
    cv::Vec3f p = internal_loc(offset+_center, ptr, _scale);

    if (!loc_valid_xy(*_points, {p[0], p[1]}))
        return {-1,-1,-1};

    return at_int((*_points), {p[0],p[1]});
}

cv::Vec3f QuadSurface::loc(const cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    return nominal_loc(offset, ptr, _scale);
}

cv::Vec3f QuadSurface::loc_raw(const cv::Vec3f &ptr)
{
    return internal_loc(_center, ptr, _scale);
}

cv::Size QuadSurface::size()
{
    ensureLoaded();
    return {static_cast<int>(_points->cols / _scale[0]), static_cast<int>(_points->rows / _scale[1])};
}

cv::Size QuadSurface::gridSize() const
{
    return {_bounds.width + 1, _bounds.height + 1};
}

cv::Vec2f QuadSurface::scale() const
{
    return _scale;
}

cv::Vec3f QuadSurface::center() const
{
    return _center;
}

vcr::grid::SurfaceOptions QuadSurface::gridOptions() const
{
    vcr::grid::SurfaceOptions options;
    options.components = _components;
    return options;
}

vcr::grid::Raster QuadSurface::gridRaster(double u0, double v0, double scale, cv::Size size) const
{
    const cv::Vec2d origin = surfaceToGrid({u0, v0});
    return {origin[0], origin[1], double(_scale[0]) / scale, double(_scale[1]) / scale, size.width, size.height};
}

cv::Vec2d QuadSurface::surfaceToGrid(const cv::Vec2d& surface) const
{
    return {
        (surface[0] + static_cast<double>(_center[0]))
            * static_cast<double>(_scale[0]),
        (surface[1] + static_cast<double>(_center[1]))
            * static_cast<double>(_scale[1]),
    };
}

void QuadSurface::shiftSurfaceOrigin(const cv::Vec2d& delta)
{
    // ensureLoaded() first: lazy loading overwrites _center, which would
    // silently drop a shift applied before the points materialize.
    ensureLoaded();
    if (!std::isfinite(delta[0]) || !std::isfinite(delta[1])) {
        return;
    }
    _center[0] += static_cast<float>(delta[0]);
    _center[1] += static_cast<float>(delta[1]);
}

cv::Vec2d QuadSurface::gridToSurface(const cv::Vec2d& grid) const
{
    if (!std::isfinite(_scale[0]) || !std::isfinite(_scale[1])
        || _scale[0] <= 0.0f || _scale[1] <= 0.0f) {
        return {NAN, NAN};
    }
    return {
        grid[0] / static_cast<double>(_scale[0])
            - static_cast<double>(_center[0]),
        grid[1] / static_cast<double>(_scale[1])
            - static_cast<double>(_center[1]),
    };
}

QuadSurface::SurfaceSample QuadSurface::sampleAtSurface(
    const cv::Vec2d& surface) const
{
    SurfaceSample result;
    if (!std::isfinite(surface[0]) || !std::isfinite(surface[1])) {
        result.status = SurfaceSample::Status::NonFinite;
        return result;
    }
    if (!std::isfinite(_scale[0]) || !std::isfinite(_scale[1])
        || _scale[0] <= 0.0f || _scale[1] <= 0.0f) {
        result.status = SurfaceSample::Status::InvalidScale;
        return result;
    }

    const_cast<QuadSurface*>(this)->ensureLoaded();
    if (!_points || _points->cols < 2 || _points->rows < 2) {
        result.status = SurfaceSample::Status::MissingPoints;
        return result;
    }

    result.grid = surfaceToGrid(surface);
    const double maxCol = static_cast<double>(_points->cols - 1);
    const double maxRow = static_cast<double>(_points->rows - 1);
    constexpr double gridTolerance = 1e-6;
    if (result.grid[0] < -gridTolerance || result.grid[1] < -gridTolerance
        || result.grid[0] > maxCol + gridTolerance
        || result.grid[1] > maxRow + gridTolerance) {
        result.status = SurfaceSample::Status::OutsideGrid;
        return result;
    }
    result.grid[0] = std::clamp(result.grid[0], 0.0, maxCol);
    result.grid[1] = std::clamp(result.grid[1], 0.0, maxRow);

    const int col = std::min(
        static_cast<int>(std::floor(result.grid[0])), _points->cols - 2);
    const int row = std::min(
        static_cast<int>(std::floor(result.grid[1])), _points->rows - 2);
    const cv::Vec3f& p00 = (*_points)(row, col);
    const cv::Vec3f& p01 = (*_points)(row, col + 1);
    const cv::Vec3f& p10 = (*_points)(row + 1, col);
    const cv::Vec3f& p11 = (*_points)(row + 1, col + 1);
    if (!isValidPointSample(p00) || !isValidPointSample(p01)
        || !isValidPointSample(p10) || !isValidPointSample(p11)) {
        result.status = SurfaceSample::Status::InvalidQuad;
        return result;
    }

    const float colFraction =
        static_cast<float>(result.grid[0] - static_cast<double>(col));
    const float rowFraction =
        static_cast<float>(result.grid[1] - static_cast<double>(row));
    const cv::Vec3f top = p00 * (1.0f - colFraction) + p01 * colFraction;
    const cv::Vec3f bottom = p10 * (1.0f - colFraction) + p11 * colFraction;
    result.volume = top * (1.0f - rowFraction) + bottom * rowFraction;
    if (!isValidPointSample(result.volume)) {
        result.status = SurfaceSample::Status::NonFinite;
        return result;
    }
    result.status = SurfaceSample::Status::Valid;
    return result;
}

cv::Vec2f QuadSurface::ptrToGrid(const cv::Vec3f& ptr) const
{
    return {ptr[0] + _center[0] * _scale[0],
            ptr[1] + _center[1] * _scale[1]};
}

cv::Vec3f QuadSurface::gridToPtr(const cv::Vec2f& grid) const
{
    return {grid[0] - _center[0] * _scale[0],
            grid[1] - _center[1] * _scale[1], 0};
}

cv::Vec3f QuadSurface::normal(const cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    ensureLoaded();
    cv::Vec3f p = internal_loc(offset+_center, ptr, _scale);
    return grid_normal((*_points), p);
}

cv::Vec3f QuadSurface::gridNormal(int row, int col) const
{
    const_cast<QuadSurface*>(this)->ensureLoaded();
    if (!_points) {
        return {NAN, NAN, NAN};
    }
    if (row < 0 || row >= _points->rows || col < 0 || col >= _points->cols)
        return {NAN, NAN, NAN};

    // Recompute on every call instead of caching the full grid (~48 MB
    // per surface). grid_normal() is cheap (bilinear tangents + cross).
    if ((*_points)(row, col)[0] == -1.f) {
        return {NAN, NAN, NAN};
    }
    return grid_normal(*_points, cv::Vec3f(static_cast<float>(col),
                                           static_cast<float>(row),
                                           0.0f));
}

void QuadSurface::setChannel(const std::string& name, const cv::Mat& channel)
{
    _channels[name] = channel;
}

cv::Mat QuadSurface::channel(const std::string& name, int flags)
{
    ensureLoaded();
    if (_channels.count(name)) {
        cv::Mat& channel = _channels[name];
        if (channel.empty() && _channelLoader) {
            channel = _channelLoader(name);
        } else if (channel.empty()) {
            // On-demand loading
            std::filesystem::path channel_path = path / (name + ".tif");
            if (std::filesystem::exists(channel_path)) {
                std::vector<cv::Mat> layers;
                // OpenCV rejects some sample layouts (e.g. two float components).
                if (readSamplesTiffFirst(channel_path)) {
                    channel = readChannelSamples(channel_path);
                } else {
                    cv::imreadmulti(channel_path.string(), layers, cv::IMREAD_UNCHANGED);
                    if (!layers.empty()) {
                        channel = layers[0];
                    }
                }
            }
        }

        if (name == "mask") {
            normalizeMaskChannel(channel);
        }

        if (!channel.empty() && !(flags & SURF_CHANNEL_NORESIZE)) {
            cv::Mat scaled_channel;
            cv::resize(channel, scaled_channel, _points->size(), 0, 0, cv::INTER_NEAREST);
            if (name == "mask") {
                normalizeMaskChannel(scaled_channel);
            }
            return scaled_channel;
        }
        return channel;
    }
    return cv::Mat();
}

std::vector<std::string> QuadSurface::channelNames() const
{
    std::vector<std::string> names;
    for (const auto& pair : _channels) {
        names.push_back(pair.first);
    }
    return names;
}

bool QuadSurface::channelLoaded(const std::string& name) const
{
    const auto found = _channels.find(name);
    return found != _channels.end() && !found->second.empty();
}

int QuadSurface::countValidPoints() const
{
    const_cast<QuadSurface*>(this)->ensureLoaded();
    if (!_points) return 0;
    auto range = validPoints();
    return static_cast<int>(std::distance(range.begin(), range.end()));
}

int QuadSurface::countValidQuads() const
{
    const_cast<QuadSurface*>(this)->ensureLoaded();
    if (!_points) return 0;
    auto range = validQuads();
    return static_cast<int>(std::distance(range.begin(), range.end()));
}

void QuadSurface::unloadCaches()
{
    {
        std::lock_guard<std::mutex> cacheLock(_cacheMutex);
        _validMaskCache = cv::Mat_<uint8_t>();
    }
    // Release loaded channel pixel data but keep the keys so channel(name)
    // still knows which channels exist on disk and can lazy-reload them.
    for (auto& [_, mat] : _channels) {
        mat.release();
    }
}

cv::Mat_<uint8_t> QuadSurface::validMask() const
{
    const_cast<QuadSurface*>(this)->ensureLoaded();
    if (!_points || _points->empty()) {
        return cv::Mat_<uint8_t>();
    }

    // Callers may ask from several threads; the lazy build is guarded.
    std::lock_guard<std::mutex> cacheLock(_cacheMutex);

    if (!_validMaskCache.empty() &&
        _validMaskCache.rows == _points->rows &&
        _validMaskCache.cols == _points->cols) {
        return _validMaskCache;
    }

    const int rows = _points->rows;
    const int cols = _points->cols;
    cv::Mat_<uint8_t> mask(rows, cols);

    // A vertex is invalid when a coordinate is NaN or it is the (-1, -1, -1) sentinel.
#pragma omp parallel for schedule(dynamic, 16)
    for (int j = 0; j < rows; ++j) {
        const cv::Vec3f* __restrict__ row = (*_points)[j];
        uint8_t* __restrict__ dst = mask[j];
        for (int i = 0; i < cols; ++i) {
            const float a = row[i][0], b = row[i][1], c = row[i][2];
            const int nan = (a != a) | (b != b) | (c != c);
            const int sent = (a == -1.f) & (b == -1.f) & (c == -1.f);
            dst[i] = (nan | sent) ? uint8_t(0) : uint8_t(255);
        }
    }
    _validMaskCache = mask;
    return mask;
}

void QuadSurface::writeValidMask(const cv::Mat& img)
{
    if (path.empty()) {
        return;
    }
    std::lock_guard<std::recursive_mutex> dirLock(dirWriteMutex(path));
    std::filesystem::path maskPath = path / "mask.tif";
    cv::Mat_<uint8_t> mask = validMask();

    if (img.empty()) {
        writeTiff(maskPath, mask, -1, 1024, 1024, -1.0f, COMPRESSION_LZW, dpi_);
    } else {
        std::vector<cv::Mat> layers = {mask, img};
        cv::imwritemulti(maskPath.string(), layers);
    }
}

void QuadSurface::invalidateCache()
{
    if (_points) {
        _bounds = {0, 0, _points->cols - 1, _points->rows - 1};
        _center = surfaceCenterFor(*_points, _scale);
    } else {
        _bounds = {0, 0, -1, -1};
        _center = {0, 0, 0};
    }

    _bbox = {{-1, -1, -1}, {-1, -1, -1}};
    std::lock_guard<std::mutex> cacheLock(_cacheMutex);
    _validMaskCache = cv::Mat_<uint8_t>();
}

static inline float sdist(const cv::Vec3f &a, const cv::Vec3f &b)
{
    cv::Vec3f d = a-b;
    // return d.dot(d);
    return d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
}

static inline cv::Vec2f mul(const cv::Vec2f &a, const cv::Vec2f &b)
{
    return{a[0]*b[0],a[1]*b[1]};
}

static bool pointSearchAxisContains(float value, int size)
{
    // Narrow ribbons have no interior margin, but still contain bilinear cells.
    if (size <= 3)
        return size >= 2 && value >= 0 && value < size - 1;
    const int rounded = cvRound(value);
    return rounded >= 1 && rounded < size - 1;
}

static float pointSearchAxisSeed(int size)
{
    return size <= 3 ? 0.5f * (size - 1) : float(1 + rand() % (size - 3));
}

template <typename E>
static float search_min_loc(const cv::Mat_<E> &points, cv::Vec2f &loc, cv::Vec3f &out, cv::Vec3f tgt, cv::Vec2f init_step, float min_step_x)
{
    const auto contains = [&](const cv::Vec2f& p) {
        return pointSearchAxisContains(p[0], points.cols) &&
               pointSearchAxisContains(p[1], points.rows);
    };
    if (!contains(loc)) {
        out = {-1,-1,-1};
        return -1;
    }

    bool changed = true;
    E val = at_int(points, loc);
    out = val;
    float best = sdist(val, tgt);
    float res;

    // 8-neighbor search offsets: hold in a static constexpr-ish array
    // instead of allocating a fresh vector per call. pointTo() calls
    // this up to ~10*max_iters times per navigation click.
    static const std::array<cv::Vec2f, 8> search = {
        cv::Vec2f{0,-1}, cv::Vec2f{0,1},
        cv::Vec2f{-1,-1}, cv::Vec2f{-1,0}, cv::Vec2f{-1,1},
        cv::Vec2f{1,-1},  cv::Vec2f{1,0},  cv::Vec2f{1,1}};
    cv::Vec2f step = init_step;

    while (changed) {
        changed = false;

        for(auto &off : search) {
            cv::Vec2f cand = loc+mul(off,step);

            //just skip if out of bounds
            if (!contains(cand))
                continue;

            val = at_int(points, cand);
            res = sdist(val, tgt);
            if (res < best) {
                changed = true;
                best = res;
                loc = cand;
                out = val;
            }
        }

        if (changed)
            continue;

        step *= 0.5;
        changed = true;

        if (step[0] < min_step_x)
            break;
    }

    return sqrt(best);
}


//search the surface point that is closest to th tgt coord
template <typename E>
static float pointTo_(cv::Vec2f &loc, const cv::Mat_<E> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale)
{
    if (points.cols < 2 || points.rows < 2)
        return -1;
    loc = cv::Vec2f(points.cols/2,points.rows/2);
    cv::Vec3f _out;

    cv::Vec2f step_small = {std::max(1.0f,scale),std::max(1.0f,scale)};
    float min_mul = std::min(0.1*points.cols/scale,0.1*points.rows/scale);
    cv::Vec2f step_large = {min_mul*scale,min_mul*scale};

    float dist = search_min_loc(points, loc, _out, tgt, step_small, scale*0.1);

    if (dist < th && dist >= 0) {
        return dist;
    }

    cv::Vec2f min_loc = loc;
    float min_dist = dist;
    if (min_dist < 0)
        min_dist = 10*(points.cols/scale+points.rows/scale);

    //FIXME is this excessive?
    int r_full = 0;
    for(int r=0;r<10*max_iters && r_full < max_iters;r++) {
        //FIXME skipn invalid init locs!
        loc = {pointSearchAxisSeed(points.cols), pointSearchAxisSeed(points.rows)};

        if (points(loc[1],loc[0])[0] == -1)
            continue;

        r_full++;

        float dist = search_min_loc(points, loc, _out, tgt, step_large, scale*0.1);

        if (dist < th && dist >= 0) {
            dist = search_min_loc(points, loc, _out, tgt, step_small, scale*0.1);
            return dist;
        } else if (dist >= 0 && dist < min_dist) {
            min_loc = loc;
            min_dist = dist;
        }
    }

    loc = min_loc;
    return min_dist;
}

float pointTo(cv::Vec2f &loc, const cv::Mat_<cv::Vec3d> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale)
{
    return pointTo_(loc, points, tgt, th, max_iters, scale);
}

float pointTo(cv::Vec2f &loc, const cv::Mat_<cv::Vec3f> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale)
{
    return pointTo_(loc, points, tgt, th, max_iters, scale);
}

float lookupDepthIndex(QuadSurface* surface, int row, int col)
{
    if (!surface) {
        return NAN;
    }
    cv::Mat dChannel = surface->channel("d");
    if (dChannel.empty()) {
        return NAN;
    }
    if (row < 0 || row >= dChannel.rows || col < 0 || col >= dChannel.cols) {
        return NAN;
    }
    if (dChannel.type() == CV_32F) {
        return dChannel.at<float>(row, col);
    }
    if (dChannel.type() == CV_64F) {
        return static_cast<float>(dChannel.at<double>(row, col));
    }
    return NAN;
}

//search the surface point that is closest to the target coord
float QuadSurface::pointTo(cv::Vec3f &ptr, const cv::Vec3f &tgt, float th, int max_iters,
                           SurfacePatchIndex* surfaceIndex)
{
    ensureLoaded();
    if (_points->cols < 2 || _points->rows < 2)
        return -1;
    cv::Vec2f loc = cv::Vec2f(ptr[0], ptr[1]) + cv::Vec2f(_center[0]*_scale[0], _center[1]*_scale[1]);
    cv::Vec3f _out;

    cv::Vec2f step_small = {std::max(1.0f,_scale[0]), std::max(1.0f,_scale[1])};
    float min_mul = std::min(0.1*_points->cols/_scale[0], 0.1*_points->rows/_scale[1]);
    cv::Vec2f step_large = {min_mul*_scale[0], min_mul*_scale[1]};

    // Try initial location first
    float dist = search_min_loc(*_points, loc, _out, tgt, step_small, _scale[0]*0.1);

    if (dist < th && dist >= 0) {
        ptr = cv::Vec3f(loc[0], loc[1], 0) - cv::Vec3f(_center[0]*_scale[0], _center[1]*_scale[1], 0);
        return dist;
    }

    cv::Vec2f min_loc = loc;
    float min_dist = dist;
    if (min_dist < 0)
        min_dist = 10*(_points->cols/_scale[0]+_points->rows/_scale[1]);

    // Try accelerated search using spatial indices
    if (surfaceIndex && !surfaceIndex->empty()) {
        SurfacePatchIndex::SurfacePtr targetSurface(this, [](QuadSurface*) {});
        SurfacePatchIndex::PointQuery query;
        query.worldPoint = tgt;
        query.tolerance = th;
        query.surfaces.only = targetSurface;
        if (auto hit = surfaceIndex->locate(query)) {
            ptr = hit->ptr;
            return hit->distance;
        }

        return std::nextafter(th, std::numeric_limits<float>::infinity());
    }

    // Fall back to random sampling if the index didn't help
    int r_full = 0;
    int skip_count = 0;
    for(int r=0; r<10*max_iters && r_full<max_iters; r++) {
        loc = {pointSearchAxisSeed(_points->cols), pointSearchAxisSeed(_points->rows)};

        if ((*_points)(loc[1],loc[0])[0] == -1) {
            skip_count++;
            if (skip_count > max_iters / 10) {
                cv::Vec2f dir = { (float)(rand() % 3 - 1), (float)(rand() % 3 - 1) };
                if (dir[0] == 0 && dir[1] == 0) dir = {1, 0};

                cv::Vec2f current_pos = loc;
                bool first_valid_found = false;
                for (int i = 0; i < std::max(_points->cols, _points->rows); ++i) {
                    current_pos += dir;
                    if (current_pos[0] < 1 || current_pos[0] >= _points->cols - 1 ||
                        current_pos[1] < 1 || current_pos[1] >= _points->rows - 1) {
                        break; // Reached border
                    }

                    if ((*_points)((int)current_pos[1], (int)current_pos[0])[0] != -1) {
                        if (first_valid_found) {
                            loc = current_pos;
                            break; // Found second consecutive valid point
                        }
                        first_valid_found = true;
                    } else {
                        first_valid_found = false;
                    }
                }
                if ((*_points)(loc[1],loc[0])[0] == -1) continue; // if we didn't find a valid point
            } else {
                continue;
            }
        }

        r_full++;

        float dist = search_min_loc(*_points, loc, _out, tgt, step_large, _scale[0]*0.1);

        if (dist < th && dist >= 0) {
            dist = search_min_loc((*_points), loc, _out, tgt, step_small, _scale[0]*0.1);
            ptr = cv::Vec3f(loc[0], loc[1], 0) - cv::Vec3f(_center[0]*_scale[0], _center[1]*_scale[1], 0);
            return dist;
        } else if (dist >= 0 && dist < min_dist) {
            min_loc = loc;
            min_dist = dist;
        }
    }

    ptr = cv::Vec3f(min_loc[0], min_loc[1], 0) - cv::Vec3f(_center[0]*_scale[0], _center[1]*_scale[1], 0);
    return min_dist;
}

std::string QuadSurface::objectUuid() const
{
    if (!meta.is_object() || !meta.contains("uuid") || !meta["uuid"].is_string())
        return {};
    return meta["uuid"].get_string();
}

std::string QuadSurface::objectUuidOrNew() const
{
    std::string uuid = objectUuid();
    return uuid.empty() ? vc::objects::util::newUuid() : uuid;
}

void QuadSurface::save(const std::filesystem::path &path_, bool force_overwrite)
{
    saveWithUuid(path_, objectUuidOrNew(), force_overwrite);
}

void QuadSurface::saveAsNew(const std::filesystem::path &path_, bool force_overwrite)
{
    saveWithUuid(path_, vc::objects::util::newUuid(), force_overwrite);
}

void QuadSurface::saveEdited(const std::filesystem::path &source,
                             const std::filesystem::path &dest,
                             bool force_overwrite)
{
    std::error_code ec;
    if (std::filesystem::exists(dest, ec) && std::filesystem::equivalent(source, dest, ec))
        save(dest, force_overwrite);
    else
        saveAsNew(dest, force_overwrite);
}

std::recursive_mutex& QuadSurface::dirWriteMutex(const std::filesystem::path& dir)
{
    return surfaceDirWriteMutex(dir);
}

void QuadSurface::saveOverwrite()
{
    if (path.empty()) {
        throw std::runtime_error("QuadSurface::saveOverwrite() requires a valid path");
    }
    std::lock_guard<std::recursive_mutex> dirLock(dirWriteMutex(path));

    std::filesystem::path final_path = path;
    const std::string uuid = objectUuidOrNew();

    // Snapshot the on-disk state before overwriting. saveSnapshot() writes
    // a rotating backup at <parent>/backups/<segname>/{0..N-1}/ so a
    // destructive save (e.g. corrupted in-memory _points) is recoverable.
    // Failures (disk full, permissions) are logged but never block the
    // user's edit — the backup is best-effort.
    try {
        saveSnapshot();  // configured count, throttled
    } catch (const std::exception& e) {
        Logger()->warn("saveOverwrite: snapshot failed for {}: {}",
                       final_path.string(), e.what());
    }

    saveWithUuid(final_path, uuid, true);
    path = final_path;
    id = final_path.filename().string();
}

void QuadSurface::invalidateMask()
{
    // Clear from memory
    _channels.erase("mask");
    {
        std::lock_guard<std::mutex> cacheLock(_cacheMutex);
        _validMaskCache = cv::Mat_<uint8_t>();
    }

    // Delete from disk
    if (!path.empty()) {
        std::filesystem::path maskFile = path / "mask.tif";
        std::error_code ec;
        std::filesystem::remove(maskFile, ec);
    }
}

// Single-channel 8U/16U/32F -> untiled uncompressed TIFF; else cv::imwrite.
void QuadSurface::writeChannelFile(const std::filesystem::path& dir, const std::string& name, const cv::Mat& mat)
{
    bool wrote = false;
    if (mat.channels() == 1 &&
        (mat.type() == CV_8UC1 || mat.type() == CV_16UC1 || mat.type() == CV_32FC1))
    {
        try {
            writeTiff(dir / (name + ".tif"), mat, -1, 0, 0, -1.0f, COMPRESSION_NONE, dpi_);
            wrote = true;
        } catch (...) {
            wrote = false; // Fall back to OpenCV
        }
    }

    // OpenCV writes 1, 3 or 4 channels; others (e.g. the two-component uv) keep their samples.
    if (!wrote && (mat.channels() == 2 || mat.channels() > 4)) {
        writeChannelSamples(dir / (name + ".tif"), mat);
        wrote = true;
    }
    if (!wrote) {
        std::vector<int> compression_params = { cv::IMWRITE_TIFF_COMPRESSION, COMPRESSION_NONE };
        cv::imwrite((dir / (name + ".tif")).string(), mat, compression_params);
    }
}

void QuadSurface::writeDataToDirectory(const std::filesystem::path& dir, const std::string& skipChannel)
{
    // Split the points matrix into x, y, z channels
    std::vector<cv::Mat> xyz;
    cv::split((*_points), xyz);

    // QuadSurface tifxyz files are written as untiled, uncompressed TIFFs.
    writeTiff(dir / "x.tif", xyz[0], -1, 0, 0, -1.0f, COMPRESSION_NONE, dpi_);
    writeTiff(dir / "y.tif", xyz[1], -1, 0, 0, -1.0f, COMPRESSION_NONE, dpi_);
    writeTiff(dir / "z.tif", xyz[2], -1, 0, 0, -1.0f, COMPRESSION_NONE, dpi_);

    // Save additional channels
    for (auto const& [name, mat] : _channels) {
        if (!mat.empty() && (skipChannel.empty() || name != skipChannel)) {
            writeChannelFile(dir, name, mat);
        }
    }
}

// Minimum wall-clock gap between rotating snapshots of the same segment.
// saveOverwrite()/autosave/growth all snapshot; without this, edit-cadence
// saves would copy the whole segment dir into backups/ many times a second.
static constexpr std::chrono::seconds kSnapshotThrottle{120};

// Rotating snapshots kept per segment when the caller does not say.
static constexpr int kDefaultBackupCount = 10;

void QuadSurface::saveSnapshot(int maxBackups, bool force)
{
    if (path.empty()) {
        throw std::runtime_error("QuadSurface::saveSnapshot() requires a valid path");
    }
    std::lock_guard<std::recursive_mutex> dirLock(dirWriteMutex(path));
    if (maxBackups < 0) {
        maxBackups = kDefaultBackupCount;
    }
    if (maxBackups <= 0) {
        return;  // backups disabled
    }

    // No on-disk state to back up yet — the first save will populate
    // the directory; subsequent saveOverwrite calls will pick up the
    // rolling history from then on. Treat this as a no-op rather than
    // an error so the saveOverwrite call site doesn't have to special
    // case it.
    if (!std::filesystem::exists(path / "x.tif")) {
        return;
    }

    // Backups live in a "backups/" dir beside the segment.
    std::filesystem::path backupsDir = path.parent_path() / "backups" / path.filename();

    std::error_code ec;
    std::filesystem::create_directories(backupsDir, ec);
    if (ec) {
        throw std::runtime_error("Failed to create backups directory: " + ec.message());
    }

    // Find existing backup directories and determine next backup number.
    // Also track the newest slot's mtime so we can throttle: many operations
    // (autosave, push-pull edits, growth steps) snapshot via saveOverwrite,
    // which at edit cadence would copy the whole segment dir multiple times a
    // second. Skip if a backup was taken within the throttle window.
    std::vector<int> existingBackups;
    std::filesystem::file_time_type newestBackup{};
    bool haveNewest = false;
    if (std::filesystem::exists(backupsDir)) {
        for (const auto& entry : std::filesystem::directory_iterator(backupsDir)) {
            if (entry.is_directory()) {
                try {
                    int backupNum = std::stoi(entry.path().filename().string());
                    if (backupNum >= 0 && backupNum < maxBackups) {
                        existingBackups.push_back(backupNum);
                        std::error_code mtEc;
                        const auto mt = std::filesystem::last_write_time(entry.path(), mtEc);
                        if (!mtEc && (!haveNewest || mt > newestBackup)) {
                            newestBackup = mt;
                            haveNewest = true;
                        }
                    }
                } catch (...) {
                    // Skip non-numeric directories
                }
            }
        }
    }

    // Throttle: at most one snapshot per segment per kSnapshotThrottle.
    if (!force && haveNewest) {
        const auto age = std::filesystem::file_time_type::clock::now() - newestBackup;
        if (age < kSnapshotThrottle) {
            return;
        }
    }

    std::sort(existingBackups.begin(), existingBackups.end());

    std::filesystem::path snapshot_dest;

    if (existingBackups.size() < static_cast<size_t>(maxBackups)) {
        // We have room for more backups, find the first available slot
        int nextBackup = 0;
        for (int i = 0; i < maxBackups; ++i) {
            if (std::find(existingBackups.begin(), existingBackups.end(), i) == existingBackups.end()) {
                nextBackup = i;
                break;
            }
        }
        snapshot_dest = backupsDir / std::to_string(nextBackup);
    } else {
        // We're at the limit, rotate backups
        // Delete the oldest (0)
        std::filesystem::remove_all(backupsDir / "0", ec);

        // Rename all backups down by 1 (1->0, 2->1, etc.)
        for (int i = 1; i < maxBackups; ++i) {
            std::filesystem::path oldPath = backupsDir / std::to_string(i);
            std::filesystem::path newPath = backupsDir / std::to_string(i - 1);
            if (std::filesystem::exists(oldPath)) {
                std::filesystem::rename(oldPath, newPath, ec);
            }
        }

        // New backup goes in the last slot
        snapshot_dest = backupsDir / std::to_string(maxBackups - 1);
    }

    // Create the snapshot directory
    std::filesystem::create_directories(snapshot_dest, ec);
    if (ec) {
        throw std::runtime_error("Failed to create snapshot directory: " + ec.message());
    }

    // Capture the LAST PERSISTED state — copy every regular file in
    // path/ verbatim into the snapshot slot. This preserves x/y/z.tif,
    // mask.tif, ancillary channels, meta.json, corrections.json, etc.
    // without re-serializing in-memory data.
    //
    // The previous implementation called writeDataToDirectory(...)
    // which serialized the current in-memory _points. That broke
    // recovery for the saveOverwrite case: when the in-memory state
    // is what we're trying to roll back FROM, persisting it as the
    // backup snapshot leaves no good state to recover.
    //
    // Subdirectories (e.g. .tmp_* from in-flight saves) are skipped.
    for (const auto& entry : std::filesystem::directory_iterator(path, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) {
            continue;
        }
        std::error_code copyEc;
        std::filesystem::copy_file(
            entry.path(),
            snapshot_dest / entry.path().filename(),
            std::filesystem::copy_options::overwrite_existing,
            copyEc);
        if (copyEc) {
            Logger()->warn("saveSnapshot: copy failed for {}: {}",
                           entry.path().string(), copyEc.message());
        }
    }
    if (ec) {
        Logger()->warn("saveSnapshot: directory iteration failed for {}: {}",
                       path.string(), ec.message());
    }

    // e.g. "autosaved seg_1234/3 -> /path/to/paths/backups/seg_1234/3"
    Logger()->info("autosaved {}/{} -> {}",
                   path.filename().string(),
                   snapshot_dest.filename().string(),
                   snapshot_dest.string());
}


void QuadSurface::saveWithUuid(const std::filesystem::path &path_, const std::string &uuid, bool force_overwrite)
{
    std::filesystem::path target_path = path_;
    std::filesystem::path final_path = path_;

    // Serialize on the TARGET dir (may differ from this->path for new-dir saves)
    // so the atomic directory swap can't race another writer to the same dir.
    std::lock_guard<std::recursive_mutex> dirLock(dirWriteMutex(final_path));

    if (!force_overwrite && std::filesystem::exists(final_path))
        throw std::runtime_error("path already exists!");

    // Save to temporary location first for atomic operation.
    // Use a unique suffix so concurrent saves to the same path don't collide.
    auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
    std::string tmp_dir_name = ".tmp_" + std::to_string(tid);
    std::filesystem::path temp_path = target_path.parent_path() / tmp_dir_name / target_path.filename();
    std::filesystem::create_directories(temp_path.parent_path());

    // Temporarily set path for any operations that might need it
    std::filesystem::path original_path = path;
    path = temp_path;

    if (!std::filesystem::create_directories(path)) {
        if (!std::filesystem::exists(path)) {
            path = original_path; // Restore on error
            throw std::runtime_error("error creating dir for QuadSurface::save(): " + path.string());
        }
    }

    // Write surface data to temp directory
    try {
        writeDataToDirectory(path);
    } catch (const std::exception& e) {
        path = original_path; // Restore on error
        throw;
    }

    // Prepare and write metadata
    {
        // The cached _bbox can be stale here: it is seeded from meta.json at
        // load time, and the point grid can be mutated directly afterwards
        // (rawPointsPtr(), validPoints(), ...) without any invalidation hook.
        // Drop the cache so bbox() recomputes from the points actually being
        // saved (#1272).
        _bbox = {{-1, -1, -1}, {-1, -1, -1}};
        auto lo = utils::Json::array();
        lo.push_back(bbox().low[0]); lo.push_back(bbox().low[1]); lo.push_back(bbox().low[2]);
        auto hi = utils::Json::array();
        hi.push_back(bbox().high[0]); hi.push_back(bbox().high[1]); hi.push_back(bbox().high[2]);
        auto bb = utils::Json::array();
        bb.push_back(std::move(lo)); bb.push_back(std::move(hi));
        meta["bbox"] = std::move(bb);
    }
    meta["type"] = "seg";
    meta["uuid"] = uuid;
    meta["format"] = "tifxyz";
    {
        auto sc = utils::Json::array();
        sc.push_back(_scale[0]); sc.push_back(_scale[1]);
        meta["scale"] = std::move(sc);
    }

    std::ofstream o(path / "meta.json.tmp");
    o << meta.dump(4) << std::endl;
    o.close();

    // Rename to make creation atomic
    std::filesystem::rename(path / "meta.json.tmp", path / "meta.json");

    // Preserve auxiliary files from the existing directory before replacing.
    // Lazy-loaded channels are tracked by name with an empty cv::Mat; if a
    // geometry-only save skips serializing one of those channels, carry the
    // existing file forward instead of dropping it from the replaced directory.
    if (force_overwrite && std::filesystem::exists(final_path)) {
        std::error_code iterEc;
        for (const auto& entry : std::filesystem::directory_iterator(final_path, iterEc)) {
            if (iterEc) {
                break;
            }
            if (!entry.is_regular_file()) {
                continue;
            }

            const std::filesystem::path dst = temp_path / entry.path().filename();
            if (std::filesystem::exists(dst)) {
                continue;
            }

            std::error_code copyEc;
            std::filesystem::copy_file(entry.path(), dst,
                std::filesystem::copy_options::overwrite_existing, copyEc);
            // Ignore errors for auxiliary files; x/y/z/meta were just written
            // and remain the integrity-critical parts of the surface save.
        }
    }

    // Atomically move the saved data to the final location
    bool replacedExisting = false;
    if (force_overwrite && std::filesystem::exists(final_path)) {
#ifdef __linux__
        if (renameat2(AT_FDCWD, temp_path.c_str(), AT_FDCWD, final_path.c_str(), RENAME_EXCHANGE) != 0) {
            const int err = errno;
            if (err == ENOSYS || err == EINVAL) {
                // System doesn't support atomic exchange, fall back to remove + rename
                std::filesystem::remove_all(final_path);
                std::filesystem::rename(temp_path, final_path);
                replacedExisting = true;
            } else {
                path = original_path; // Restore on error
                const std::error_code ec(err, std::generic_category());
                throw std::runtime_error("atomic exchange failed for " + temp_path.string() +
                                       " and " + final_path.string() + ": " + ec.message());
            }
        } else {
            // Atomic exchange succeeded, clean up the old data now in temp location
            std::error_code cleanupErr;
            std::filesystem::remove_all(temp_path, cleanupErr);
            if (cleanupErr) {
                path = original_path; // Restore on error
                throw std::runtime_error("failed to clean up previous segmentation data at " +
                                       temp_path.string() + ": " + cleanupErr.message());
            }
            replacedExisting = true;
        }
#elif defined(_WIN32)
        try {
            replaceDirectoryContents(temp_path, final_path);
        } catch (...) {
            path = original_path;
            throw;
        }
        replacedExisting = true;
#else
        // renameat2/RENAME_EXCHANGE is Linux-only; use remove + rename fallback
        std::filesystem::remove_all(final_path);
        std::filesystem::rename(temp_path, final_path);
        replacedExisting = true;
#endif
    }

    if (!replacedExisting) {
        std::filesystem::rename(temp_path, final_path);
    }

    // Clean up the per-thread temp parent directory if empty
    {
        std::error_code ec;
        std::filesystem::remove(temp_path.parent_path(), ec);
        // Ignore errors - directory might not be empty or might not exist
    }

    // Update the path to the final canonical location
    path = final_path;
    id = final_path.filename().string();
}

Rect3D QuadSurface::bbox()
{
    ensureLoaded();
    if (_bbox.low[0] == -1) {
        _bbox.low = (*_points)(0,0);
        _bbox.high = (*_points)(0,0);

        for(int j=0;j<_points->rows;j++)
            for(int i=0;i<_points->cols;i++)
                if (_bbox.low[0] == -1)
                    _bbox = {(*_points)(j,i),(*_points)(j,i)};
                else if ((*_points)(j,i)[0] != -1)
                    _bbox = expand_rect(_bbox, (*_points)(j,i));
    }

    return _bbox;
}

namespace {

std::unique_ptr<QuadSurface> load_quad_from_tifxyz_impl(
    const std::filesystem::path &path,
    const std::optional<cv::Rect> &requestedRegion,
    int flags)
{
    recoverIncompleteSave(path);

    cv::Rect loadedRegion;
    auto read_band_into = [&requestedRegion, &loadedRegion](
                             const std::filesystem::path& fpath,
                             cv::Mat_<cv::Vec3f>& points,
                             int channel,
                             int& outW, int& outH) -> void
    {
        TIFF* tif = TIFFOpen(fpath.string().c_str(), "r");
        if (!tif) {
            throw std::runtime_error("Failed to open TIFF: " + fpath.string());
        }

        // Basic geometry
        uint32_t W=0, H=0;
        if (!TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &W) ||
            !TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &H)) {
            TIFFClose(tif);
            throw std::runtime_error("TIFF missing width/height: " + fpath.string());
        }

        uint16_t spp = 1; TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);
        if (spp != 1) {
            TIFFClose(tif);
            throw std::runtime_error("Expected 1 sample per pixel in " + fpath.string());
        }
        uint16_t bps = 0;  TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &bps);
        uint16_t fmt = SAMPLEFORMAT_UINT; TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &fmt);
        const int bytesPer = (bps + 7) / 8;
        if (!(bps==8 || bps==16 || bps==32 || bps==64)) {
            TIFFClose(tif);
            throw std::runtime_error("Unsupported BitsPerSample in " + fpath.string());
        }

        const cv::Rect fullRegion(0, 0, static_cast<int>(W), static_cast<int>(H));
        const cv::Rect bandRegion = requestedRegion.value_or(fullRegion);
        if (bandRegion.width <= 0 || bandRegion.height <= 0
            || (bandRegion & fullRegion) != bandRegion) {
            TIFFClose(tif);
            throw std::runtime_error("Requested TIFF region is out of bounds in "
                                     + fpath.string());
        }

        // Allocate destination if first band.
        if (points.empty()) {
            points.create(bandRegion.height, bandRegion.width);
            // Initialize as invalid
            for (int y=0;y<points.rows;++y)
                for (int x=0;x<points.cols;++x)
                    points(y,x) = cv::Vec3f(-1.f,-1.f,-1.f);
            outW = static_cast<int>(W);
            outH = static_cast<int>(H);
            loadedRegion = bandRegion;
        } else {
            if (outW != static_cast<int>(W) || outH != static_cast<int>(H)
                || loadedRegion != bandRegion) {
                TIFFClose(tif);
                throw std::runtime_error("Band size mismatch in " + fpath.string());
            }
        }

        auto to_float = [fmt,bps,bytesPer](const uint8_t* p)->float {
            float v=0.f;
            switch(fmt) {
                case SAMPLEFORMAT_IEEEFP:
                    if (bps==32) { std::memcpy(&v, p, 4); return v; }
                    if (bps==64) { double d=0.0; std::memcpy(&d, p, 8); return static_cast<float>(d); }
                    break;
                case SAMPLEFORMAT_UINT:
                {
                    if (bps==8)  { uint8_t  t=*p; return static_cast<float>(t); }
                    if (bps==16) { uint16_t t; std::memcpy(&t,p,2); return static_cast<float>(t); }
                    if (bps==32) { uint32_t t; std::memcpy(&t,p,4); return static_cast<float>(t); }
                } break;
                case SAMPLEFORMAT_INT:
                {
                    if (bps==8)  { int8_t  t=*reinterpret_cast<const int8_t*>(p); return static_cast<float>(t); }
                    if (bps==16) { int16_t t; std::memcpy(&t,p,2); return static_cast<float>(t); }
                    if (bps==32) { int32_t t; std::memcpy(&t,p,4); return static_cast<float>(t); }
                } break;
                default: break;
            }
            // Last resort: treat as 32-bit float bytes
            std::memcpy(&v, p, std::min(bytesPer, (int)sizeof(float)));
            return v;
        };

        if (TIFFIsTiled(tif)) {
            uint32_t tileW=0, tileH=0;
            TIFFGetField(tif, TIFFTAG_TILEWIDTH,  &tileW);
            TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileH);
            if (tileW==0 || tileH==0) {
                TIFFClose(tif);
                throw std::runtime_error("Invalid tile geometry in " + fpath.string());
            }
            const tmsize_t tileBytes = TIFFTileSize(tif);
            std::vector<uint8_t> tileBuf(static_cast<size_t>(tileBytes));

            const uint32_t firstTileY =
                static_cast<uint32_t>(loadedRegion.y) / tileH * tileH;
            const uint32_t firstTileX =
                static_cast<uint32_t>(loadedRegion.x) / tileW * tileW;
            const uint32_t regionEndY =
                static_cast<uint32_t>(loadedRegion.y + loadedRegion.height);
            const uint32_t regionEndX =
                static_cast<uint32_t>(loadedRegion.x + loadedRegion.width);
            for (uint32_t y0=firstTileY; y0<regionEndY; y0+=tileH) {
                const uint32_t copyY0 =
                    std::max(y0, static_cast<uint32_t>(loadedRegion.y));
                const uint32_t copyY1 = std::min(
                    std::min(y0 + tileH, H), regionEndY);
                for (uint32_t x0=firstTileX; x0<regionEndX; x0+=tileW) {
                    const uint32_t copyX0 =
                        std::max(x0, static_cast<uint32_t>(loadedRegion.x));
                    const uint32_t copyX1 = std::min(
                        std::min(x0 + tileW, W), regionEndX);
                    const ttile_t tidx = TIFFComputeTile(tif, x0, y0, 0, 0);
                    tmsize_t n = TIFFReadEncodedTile(tif, tidx, tileBuf.data(), tileBytes);
                    if (n < 0) {
                        // fill with zeros on read error
                        std::fill(tileBuf.begin(), tileBuf.end(), 0);
                    }
                    for (uint32_t sourceY=copyY0; sourceY<copyY1; ++sourceY) {
                        const uint32_t ty = sourceY - y0;
                        const uint8_t* row = tileBuf.data() + (static_cast<size_t>(ty)*tileW*bytesPer);
                        for (uint32_t sourceX=copyX0; sourceX<copyX1; ++sourceX) {
                            const uint32_t tx = sourceX - x0;
                            float fv = to_float(row + static_cast<size_t>(tx)*bytesPer);
                            cv::Vec3f& dst = points(
                                static_cast<int>(sourceY) - loadedRegion.y,
                                static_cast<int>(sourceX) - loadedRegion.x);
                            dst[channel] = fv;
                        }
                    }
                }
            }
        } else {
            const tmsize_t scanBytes = TIFFScanlineSize(tif);
            std::vector<uint8_t> scanBuf(static_cast<size_t>(scanBytes));
            const uint32_t endY =
                static_cast<uint32_t>(loadedRegion.y + loadedRegion.height);
            const uint32_t endX =
                static_cast<uint32_t>(loadedRegion.x + loadedRegion.width);
            for (uint32_t y=static_cast<uint32_t>(loadedRegion.y); y<endY; ++y) {
                if (TIFFReadScanline(tif, scanBuf.data(), y, 0) != 1) {
                    std::fill(scanBuf.begin(), scanBuf.end(), 0);
                }
                const uint8_t* row = scanBuf.data();
                for (uint32_t x=static_cast<uint32_t>(loadedRegion.x); x<endX; ++x) {
                    float fv = to_float(row + static_cast<size_t>(x)*bytesPer);
                    cv::Vec3f& dst = points(
                        static_cast<int>(y) - loadedRegion.y,
                        static_cast<int>(x) - loadedRegion.x);
                    dst[channel] = fv;
                }
            }
        }
        TIFFClose(tif);
    };

    // Read meta first (scale, uuid, etc.)
    auto metadata = utils::Json::parse_file(std::filesystem::path(path)/"meta.json");
    cv::Vec2f scale = {metadata["scale"][0].get_float(), metadata["scale"][1].get_float()};

    auto points = std::make_unique<cv::Mat_<cv::Vec3f>>();
    int W=0, H=0;
    read_band_into(std::filesystem::path(path)/"x.tif", *points, 0, W, H);
    read_band_into(std::filesystem::path(path)/"y.tif", *points, 1, W, H);
    read_band_into(std::filesystem::path(path)/"z.tif", *points, 2, W, H);

    // Missing vertices by the store's validity rule.
    for (int j=0;j<points->rows;++j)
        for (int i=0;i<points->cols;++i) {
            const cv::Vec3f& p = (*points)(j,i);
            if (!vc::geometry::validSurfacePoint(p))
                (*points)(j,i) = cv::Vec3f(-1.f,-1.f,-1.f);
        }

    // Optional mask
    const std::filesystem::path maskPath = std::filesystem::path(path)/"mask.tif";
    if (!(flags & SURF_LOAD_IGNORE_MASK) && std::filesystem::exists(maskPath)) {
        TIFF* mtif = TIFFOpen(maskPath.string().c_str(), "r");
        if (mtif) {
            uint32_t mW=0, mH=0;
            TIFFGetField(mtif, TIFFTAG_IMAGEWIDTH, &mW);
            TIFFGetField(mtif, TIFFTAG_IMAGELENGTH, &mH);
            if (mW!=0 && mH!=0) {
                uint16_t bps=0, fmt=SAMPLEFORMAT_UINT;
                TIFFGetFieldDefaulted(mtif, TIFFTAG_BITSPERSAMPLE, &bps);
                TIFFGetFieldDefaulted(mtif, TIFFTAG_SAMPLEFORMAT, &fmt);
                uint16_t spp=1;
                TIFFGetFieldDefaulted(mtif, TIFFTAG_SAMPLESPERPIXEL, &spp);
                uint16_t planarConfig = PLANARCONFIG_CONTIG;
                TIFFGetFieldDefaulted(mtif, TIFFTAG_PLANARCONFIG, &planarConfig);
                const int bytesPer = (bps+7)/8;
                const uint16_t samplesPerPixel = std::max<uint16_t>(1, spp);
                const bool isPlanarSeparate = (planarConfig == PLANARCONFIG_SEPARATE);
                const size_t pixelStride = static_cast<size_t>(bytesPer) *
                                           static_cast<size_t>(isPlanarSeparate ? 1 : samplesPerPixel);
                auto computeScaleFactor = [](uint32_t maskDim, int targetDim) -> uint32_t {
                    if (maskDim == static_cast<uint32_t>(targetDim)) {
                        return 1;
                    } else if (targetDim > 0 && (maskDim % static_cast<uint32_t>(targetDim)) == 0) {
                        return maskDim / static_cast<uint32_t>(targetDim);
                    } else {
                        return 0;
                    }
                };
                const uint32_t scaleX = computeScaleFactor(mW, W);
                const uint32_t scaleY = computeScaleFactor(mH, H);
                if (scaleX != 0 && scaleY != 0) {
                    // Channel 0 encodes mask validity; later channels remain untouched.
                    const double retainThreshold = [&]{
                        switch(fmt) {
                            case SAMPLEFORMAT_UINT:
                            case SAMPLEFORMAT_INT:
                            case SAMPLEFORMAT_IEEEFP:
                                return 255.0;
                        }
                        return 255.0;
                    }();
                    auto to_valid = [fmt,bps,bytesPer,retainThreshold](const uint8_t* p)->bool{
                        switch(fmt) {
                            case SAMPLEFORMAT_IEEEFP:
                                if (bps==32) { float v; std::memcpy(&v,p,4); return v>=static_cast<float>(retainThreshold); }
                                if (bps==64) { double d; std::memcpy(&d,p,8); return d>=retainThreshold; }
                                break;
                            case SAMPLEFORMAT_UINT:
                                if (bps==8)  return (*p)>=retainThreshold;
                                if (bps==16) { uint16_t t; std::memcpy(&t,p,2); return t>=static_cast<uint16_t>(retainThreshold); }
                                if (bps==32) { uint32_t t; std::memcpy(&t,p,4); return t>=static_cast<uint32_t>(retainThreshold); }
                                break;
                            case SAMPLEFORMAT_INT:
                                if (bps==8)  { int8_t t=*reinterpret_cast<const int8_t*>(p);  return t>=static_cast<int8_t>(retainThreshold); }
                                if (bps==16) { int16_t t; std::memcpy(&t,p,2); return t>=static_cast<int16_t>(retainThreshold); }
                                if (bps==32) { int32_t t; std::memcpy(&t,p,4); return t>=static_cast<int32_t>(retainThreshold); }
                                break;
                        }
                        // default
                        return (*p)>=retainThreshold;
                    };
                    const cv::Vec3f invalidPoint(-1.f,-1.f,-1.f);
                    auto invalidate = [&](uint32_t srcX, uint32_t srcY){
                        const uint32_t dstX = (scaleX == 1) ? srcX : (srcX / scaleX);
                        const uint32_t dstY = (scaleY == 1) ? srcY : (srcY / scaleY);
                        if (dstX >= static_cast<uint32_t>(loadedRegion.x)
                            && dstX < static_cast<uint32_t>(
                                loadedRegion.x + loadedRegion.width)
                            && dstY >= static_cast<uint32_t>(loadedRegion.y)
                            && dstY < static_cast<uint32_t>(
                                loadedRegion.y + loadedRegion.height)) {
                            (*points)(static_cast<int>(dstY) - loadedRegion.y,
                                      static_cast<int>(dstX) - loadedRegion.x) = invalidPoint;
                        }
                    };
                    if (TIFFIsTiled(mtif)) {
                        uint32_t tileW=0, tileH=0;
                        TIFFGetField(mtif, TIFFTAG_TILEWIDTH,  &tileW);
                        TIFFGetField(mtif, TIFFTAG_TILELENGTH, &tileH);
                        const tmsize_t tileBytes = TIFFTileSize(mtif);
                        std::vector<uint8_t> tileBuf(static_cast<size_t>(tileBytes));
                        const uint32_t maskX0 =
                            static_cast<uint32_t>(loadedRegion.x) * scaleX;
                        const uint32_t maskY0 =
                            static_cast<uint32_t>(loadedRegion.y) * scaleY;
                        const uint32_t maskX1 = std::min(
                            mW, static_cast<uint32_t>(
                                loadedRegion.x + loadedRegion.width) * scaleX);
                        const uint32_t maskY1 = std::min(
                            mH, static_cast<uint32_t>(
                                loadedRegion.y + loadedRegion.height) * scaleY);
                        const uint32_t firstTileX = maskX0 / tileW * tileW;
                        const uint32_t firstTileY = maskY0 / tileH * tileH;
                        for (uint32_t y0=firstTileY; y0<maskY1; y0+=tileH) {
                            const uint32_t copyY0 = std::max(y0, maskY0);
                            const uint32_t copyY1 =
                                std::min(std::min(y0 + tileH, mH), maskY1);
                            for (uint32_t x0=firstTileX; x0<maskX1; x0+=tileW) {
                                const uint32_t copyX0 = std::max(x0, maskX0);
                                const uint32_t copyX1 =
                                    std::min(std::min(x0 + tileW, mW), maskX1);
                                const ttile_t tidx = TIFFComputeTile(mtif, x0, y0, 0, 0);
                                tmsize_t n = TIFFReadEncodedTile(mtif, tidx, tileBuf.data(), tileBytes);
                                if (n < 0) std::fill(tileBuf.begin(), tileBuf.end(), 0);
                                const size_t tileRowStride = static_cast<size_t>(tileW) * pixelStride;
                                for (uint32_t sourceY=copyY0; sourceY<copyY1; ++sourceY) {
                                    const uint32_t ty = sourceY - y0;
                                    const uint8_t* row = tileBuf.data() + static_cast<size_t>(ty) * tileRowStride;
                                    for (uint32_t sourceX=copyX0; sourceX<copyX1; ++sourceX) {
                                        const uint32_t tx = sourceX - x0;
                                        const uint8_t* px = row + static_cast<size_t>(tx) * pixelStride;
                                        if (!to_valid(px)) {
                                            invalidate(sourceX, sourceY);
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        const tmsize_t scanBytes = TIFFScanlineSize(mtif);
                        std::vector<uint8_t> scanBuf(static_cast<size_t>(scanBytes));
                        const uint32_t maskX0 =
                            static_cast<uint32_t>(loadedRegion.x) * scaleX;
                        const uint32_t maskY0 =
                            static_cast<uint32_t>(loadedRegion.y) * scaleY;
                        const uint32_t maskX1 = std::min(
                            mW, static_cast<uint32_t>(
                                loadedRegion.x + loadedRegion.width) * scaleX);
                        const uint32_t maskY1 = std::min(
                            mH, static_cast<uint32_t>(
                                loadedRegion.y + loadedRegion.height) * scaleY);
                        for (uint32_t y=maskY0; y<maskY1; ++y) {
                            if (TIFFReadScanline(mtif, scanBuf.data(), y, 0) != 1) {
                                std::fill(scanBuf.begin(), scanBuf.end(), 0);
                            }
                            const uint8_t* row = scanBuf.data();
                            for (uint32_t x=maskX0; x<maskX1; ++x) {
                                const uint8_t* px = row + static_cast<size_t>(x) * pixelStride;
                                if (!to_valid(px)) {
                                    invalidate(x, y);
                                }
                            }
                        }
                    }
                }
            }
            TIFFClose(mtif);
        }
    }

    auto surf = std::make_unique<QuadSurface>(points.release(), scale);
    if (!requestedRegion) surf->path = path;
    surf->id   = path.filename().string();
    surf->meta = metadata;

    // Register extra channels lazily (left as OpenCV-based on-demand load).
    if (!requestedRegion) {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            if (entry.path().extension() == ".tif") {
                std::string filename = entry.path().stem().string();
                if (filename != "x" && filename != "y" && filename != "z") {
                    surf->setChannel(filename, cv::Mat());
                }
            }
        }
    }
    return surf;
}

} // namespace

std::unique_ptr<QuadSurface> load_quad_from_tifxyz(
    const std::filesystem::path &path, int flags)
{
    return load_quad_from_tifxyz_impl(path, std::nullopt, flags);
}

std::unique_ptr<QuadSurface> load_quad_from_tifxyz_region(
    const std::filesystem::path &path, const cv::Rect &region, int flags)
{
    return load_quad_from_tifxyz_impl(path, region, flags);
}

Rect3D expand_rect(const Rect3D &a, const cv::Vec3f &p)
{
    Rect3D res = a;
    for(int d=0;d<3;d++) {
        res.low[d] = std::min(res.low[d], p[d]);
        res.high[d] = std::max(res.high[d], p[d]);
    }

    return res;
}

utils::Json bbox_to_json(const Rect3D& bbox)
{
    auto lo = utils::Json::array();
    lo.push_back(bbox.low[0]);
    lo.push_back(bbox.low[1]);
    lo.push_back(bbox.low[2]);
    auto hi = utils::Json::array();
    hi.push_back(bbox.high[0]);
    hi.push_back(bbox.high[1]);
    hi.push_back(bbox.high[2]);
    auto arr = utils::Json::array();
    arr.push_back(std::move(lo));
    arr.push_back(std::move(hi));

    return arr;
}

bool bbox_of_valid_points(const cv::Mat_<cv::Vec3f>& points, Rect3D& out)
{
    bool any = false;
    Rect3D res;
    for (int j = 0; j < points.rows; j++)
        for (int i = 0; i < points.cols; i++) {
            const cv::Vec3f& p = points(j, i);
            if (p[0] == -1)
                continue;
            if (!any) {
                res = {p, p};
                any = true;
            } else
                res = expand_rect(res, p);
        }

    if (any)
        out = res;

    return any;
}

bool intersect(const Rect3D &a, const Rect3D &b)
{
    for(int d=0;d<3;d++) {
        if (a.high[d] < b.low[d])
            return false;
        if (a.low[d] > b.high[d])
            return false;
    }

    return true;
}

void QuadSurface::rotate(float angleDeg)
{
    ensureLoaded();
    if (!_points || _points->empty() || std::abs(angleDeg) < 0.01f) {
        return;
    }

    // Compute rotation center (center of the image)
    cv::Point2f center(
        static_cast<float>(_points->cols - 1) / 2.0f,
        static_cast<float>(_points->rows - 1) / 2.0f
    );

    // Get rotation matrix
    cv::Mat rotMat = cv::getRotationMatrix2D(center, angleDeg, 1.0);

    // Calculate bounding box of rotated image
    cv::Rect2f bbox = cv::RotatedRect(center, _points->size(), angleDeg).boundingRect2f();

    // Adjust rotation matrix to translate image to center of new canvas
    rotMat.at<double>(0, 2) += bbox.width / 2.0 - center.x;
    rotMat.at<double>(1, 2) += bbox.height / 2.0 - center.y;

    // Rotate points without blending invalid sentinels into finite coordinates.
    const cv::Size dstSize = bbox.size();
    cv::Mat_<cv::Vec3f> rotatedMat =
        warpAffinePointsLinearPreservingInvalids(*_points, rotMat, dstSize);

    // Update points
    *_points = rotatedMat;

    // Rotate all channels
    for (auto& [name, channel] : _channels) {
        if (channel.empty()) continue;

        cv::Mat rotatedChannel;
        cv::Scalar borderValue = (name == "normals") ? cv::Scalar(0, 0, 0) : cv::Scalar(0, 0, 0);
        cv::warpAffine(
            channel,
            rotatedChannel,
            rotMat,
            dstSize,
            cv::INTER_LINEAR,
            cv::BORDER_CONSTANT,
            borderValue
        );
        channel = rotatedChannel;
    }

    // Invalidate cached bbox
    invalidateCache();
}

void QuadSurface::flipU()
{
    ensureLoaded();
    if (!_points || _points->empty()) {
        return;
    }

    // Flip over the U axis means reversing the rows (V direction)
    constexpr int flipCode = 0;  // 0 = flip around x-axis (vertical flip)
    cv::flip(*_points, *_points, flipCode);

    // Flip all channels the same way
    for (auto& [name, channel] : _channels) {
        if (!channel.empty()) {
            cv::flip(channel, channel, flipCode);
        }
    }

    // Invalidate derived geometry caches.
    invalidateCache();
}

void QuadSurface::flipV()
{
    ensureLoaded();
    if (!_points || _points->empty()) {
        return;
    }

    // Flip over the V axis means reversing the columns (U direction)
    constexpr int flipCode = 1;  // 1 = flip around y-axis (horizontal flip)
    cv::flip(*_points, *_points, flipCode);

    // Flip all channels the same way
    for (auto& [name, channel] : _channels) {
        if (!channel.empty()) {
            cv::flip(channel, channel, flipCode);
        }
    }

    // Invalidate derived geometry caches.
    invalidateCache();
}

// Surface overlap/containment tests
bool contains(QuadSurface& a, const cv::Vec3f& loc, int max_iters)
{
    if (!intersect(a.bbox(), {loc,loc}))
        return false;

    cv::Vec3f ptr(0, 0, 0);
    if (a.pointTo(ptr, loc, 2.0, max_iters) <= 2.0) {
        return true;
    }
    return false;
}

bool contains(QuadSurface& a, const std::vector<cv::Vec3f>& locs)
{
    for(auto& p : locs)
        if (!contains(a, p))
            return false;

    return true;
}
