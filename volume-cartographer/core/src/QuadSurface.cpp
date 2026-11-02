#include "vc/core/util/QuadSurface.hpp"
#include <vc_reader/publication.hpp>
#include "vc/core/util/ImageMat.hpp"

#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/LoadJson.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/util/SurfaceQuerySet.hpp"
#include <vc_reader/source_reader.hpp>
#include <vc_reader/tifxyz.hpp>
#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/geometry/Geometry.hpp>

#include <opencv2/imgproc.hpp>

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
#include <chrono>
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


// GCC/Clang spell the restrict qualifier __restrict__; MSVC only has __restrict.
#if defined(_MSC_VER)
#define __restrict__ __restrict
#endif

namespace {

vcr::sources::TiffOptions tiffOptions(float dpi)
{
    vcr::sources::TiffOptions options;
    if (dpi > 0) options.pixelUm = 25400.0 / dpi;
    return options;
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
                                            const nlohmann::json& metadata)
{
    // The scale as vc_reader parses it (two positive finite numbers; the file's order is
    // x, y, Metadata's rows, columns). The caller's metadata may be a partial override
    // (second constructor), so a scale it lacks comes from the on-disk meta.json.
    const auto scaleOf = [&](const vcr::zarr::Json& document) -> std::optional<cv::Vec2f> {
        if (!document.contains("scale"))
            return std::nullopt;
        const auto parsed = vcr::tifxyz::parseMetadata(document, path.string()).scale;
        return cv::Vec2f(float(parsed[1]), float(parsed[0]));
    };
    std::optional<cv::Vec2f> scale = scaleOf(metadata);
    if (!scale) {
        const auto disk = vcr::tifxyz::readMetadata(path.string()).scale;
        scale = cv::Vec2f(float(disk[1]), float(disk[0]));
    }

    // The grid's shape from x.tif alone, as a plain image source.
    vcr::sources::SourceReader xReader(vcr::sources::imageSource((path / "x.tif").string()), {});
    const auto height = xReader.shape()[0];
    const auto width = xReader.shape()[1];

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
static Rect3D rect_from_json(const nlohmann::json &json)
{
    return {{json[0][0].get<float>(),json[0][1].get<float>(),json[0][2].get<float>()},
            {json[1][0].get<float>(),json[1][1].get<float>(),json[1][2].get<float>()}};
}
} // anonymous namespace

std::vector<std::string> tifxyz_tags(const nlohmann::json& meta)
{
    std::vector<std::string> tags;
    if (!meta.is_object() || !meta.contains("tags") || !meta["tags"].is_object()) return tags;
    const auto& recorded = meta["tags"];
    for (auto it = recorded.begin(); it != recorded.end(); ++it) tags.push_back(it.key());
    return tags;
}

std::vector<std::pair<int, int>> surface_components(const nlohmann::json& meta)
{
    std::vector<std::pair<int, int>> components;
    if (meta.contains("components") && meta["components"].is_array()) {
        for (const auto& c : meta["components"]) {
            if (c.is_array() && c.size() >= 2)
                components.emplace_back(c[0].get<int>(), c[1].get<int>());
        }
    }
    return components;
}

QuadSurface::QuadSurface(const std::filesystem::path &path_)
{
    vcr::files::recoverDirectory(path_);
    path = path_;
    id = path_.filename().string();
    auto metaPath = path_ / "meta.json";
    meta = nlohmann::json::parse(std::ifstream(metaPath));

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

QuadSurface::QuadSurface(const std::filesystem::path &path_, const nlohmann::json &json)
{
    vcr::files::recoverDirectory(path_);
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

    const float xy[2] = {p[0], p[1]};
    cv::Vec3f point;
    pointSurface()->pointsAt(xy, 1, point.val, nullptr);
    if (!std::isfinite(point[0]))
        return {-1,-1,-1};
    return point;
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

vcr::grid::Vertices QuadSurface::gridVertices(std::vector<std::uint8_t>& valid, const cv::Mat_<cv::Vec3f>* points) const
{
    if (!points) points = rawPointsPtr();
    valid.assign(std::size_t(points->rows) * std::size_t(points->cols), 1);
    for (int j = 0; j < points->rows; ++j)
        for (int i = 0; i < points->cols; ++i)
        {
            const auto& p = (*points)(j, i);
            if (p == cv::Vec3f(-1.f, -1.f, -1.f) || !std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) valid[std::size_t(j) * std::size_t(points->cols) + std::size_t(i)] = 0;
        }
    return {reinterpret_cast<const float*>(points->ptr()), points->rows, points->cols, points->step1(), nullptr, valid.data()};
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
    const float xy[2] = {p[0], p[1]};
    cv::Vec3f point, normal;
    pointSurface()->pointsAt(xy, 1, point.val, normal.val);
    return normal;
}

std::shared_ptr<const vcr::grid::Surface> QuadSurface::pointSurface() const
{
    const_cast<QuadSurface*>(this)->ensureLoaded();
    std::lock_guard<std::mutex> lock(_cacheMutex);
    if (!_pointSurface) {
        std::vector<std::uint8_t> valid;
        _pointSurface = vcr::grid::Surface::snapshot(gridVertices(valid, _points.get()), gridOptions());
    }
    return _pointSurface;
}

void QuadSurface::dropPointSurface() const
{
    std::lock_guard<std::mutex> lock(_cacheMutex);
    _pointSurface.reset();
}

void QuadSurface::setChannel(const std::string& name, const cv::Mat& channel)
{
    _channels[name] = channel;
    dropPointSurface();
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
                try {
                    channel = vc::core::util::imageMat(vcr::sources::readImagePage(std::filesystem::path(channel_path).string(), 0));
                } catch (const std::exception& error) {
                    Logger()->warn("cannot read channel {}: {}", channel_path.string(), error.what());
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

void QuadSurface::writeValidMask()
{
    if (path.empty()) {
        return;
    }
    std::lock_guard<std::recursive_mutex> dirLock(dirWriteMutex(path));
    auto options = tiffOptions(dpi_);
    options.compression = "lzw";
    options.tile = std::array<int, 2>{1024, 1024};
    vcr::sources::writeTiff(path / "mask.tif", vc::core::util::imageBuffer(validMask()).view(), options);
}

void QuadSurface::invalidateCache()
{
    dropPointSurface();
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
                           SurfaceQuerySet* surfaceIndex)
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
        SurfaceQuerySet::SurfacePtr targetSurface(this, [](QuadSurface*) {});
        SurfaceQuerySet::PointQuery query;
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
    return meta["uuid"].get<std::string>();
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
    return vcr::files::directoryMutex(dir);
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

// Minimum wall-clock gap between rotating snapshots of the same segment.
// saveOverwrite()/autosave/growth all snapshot; without this, edit-cadence
// saves would copy the whole segment dir into backups/ many times a second.
static constexpr std::chrono::seconds kSnapshotThrottle{120};

// Rotating snapshots kept per segment when the caller does not say.
static constexpr int kDefaultBackupCount = 10;

void QuadSurface::saveSnapshot(int maxBackups, bool force)
{
    if (path.empty()) throw std::runtime_error("snapshot requires a surface path");
    vcr::files::snapshot(path, maxBackups < 0 ? kDefaultBackupCount : maxBackups, kSnapshotThrottle, force);
}

void QuadSurface::saveWithUuid(const std::filesystem::path& destination, const std::string& uuid, bool overwrite)
{
    ensureLoaded();
    std::lock_guard<std::recursive_mutex> lock(vcr::files::directoryMutex(destination));
    std::vector<std::uint8_t> valid;
    (void)gridVertices(valid);
    std::vector<float> xyz(_points->total() * 3);
    for (int row = 0; row < _points->rows; ++row)
        std::copy_n(reinterpret_cast<const float*>(_points->ptr(row)), _points->cols * 3,
                    xyz.data() + std::size_t(row) * _points->cols * 3);
    vcr::tifxyz::Metadata metadata;
    metadata.uuid = uuid;
    metadata.scale = {_scale[1], _scale[0]};
    metadata.extra = meta;
    metadata.extra["type"] = "seg";
    metadata.extra["format"] = "tifxyz";
    vcr::tifxyz::Surface stored(_points->rows, _points->cols, std::move(xyz), std::move(valid), std::move(metadata));
    vcr::tifxyz::WriteOptions options;
    options.overwrite = overwrite;
    options.compression = "none";
    options.tile = std::nullopt;
    options.mask = false;
    if (dpi_ > 0) options.pixelUm = 25400.0 / dpi_;
    std::vector<cv::Mat> channelStorage;
    channelStorage.reserve(_channels.size());
    for (const auto& [name, ignored] : _channels) {
        auto mat = channel(name, SURF_CHANNEL_NORESIZE);
        if (mat.empty()) continue;
        cv::Mat native;
        if (mat.channels() == 3) cv::cvtColor(mat,native,cv::COLOR_BGR2RGB);
        else if (mat.channels() == 4) cv::cvtColor(mat,native,cv::COLOR_BGRA2RGBA);
        else native = mat.isContinuous() ? mat : mat.clone();
        const char* dtype = vc::core::util::tiffDtype(native.depth());
        if (!dtype) throw std::runtime_error("unsupported surface channel type");
        channelStorage.push_back(native);
        options.channels.push_back({name,native.data,dtype,native.rows,native.cols,native.channels()});
    }
    vcr::tifxyz::write(destination,stored,options);
    const auto savedMetadata = vcr::tifxyz::readMetadata(destination.string());
    meta = savedMetadata.extra;
    if (savedMetadata.area) meta["area"] = *savedMetadata.area;
    meta["uuid"] = uuid;
    meta["scale"] = {_scale[0],_scale[1]};
    if (const auto box=stored.bounds()) {
        _bbox={{float((*box)[0]),float((*box)[1]),float((*box)[2])},
               {float((*box)[3]),float((*box)[4]),float((*box)[5])}};
        meta["bbox"]={{(*box)[0],(*box)[1],(*box)[2]},{(*box)[3],(*box)[4],(*box)[5]}};
    }
    path=destination;
    id=destination.filename().string();
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
    vcr::files::recoverDirectory(path);

    // Read meta first (scale, uuid, etc.)
    auto metadata = nlohmann::json::parse(std::ifstream(std::filesystem::path(path)/"meta.json"));
    cv::Vec2f scale = {metadata["scale"][0].get<float>(), metadata["scale"][1].get<float>()};

    vcr::tifxyz::Reader reader(path.string(), !(flags & SURF_LOAD_IGNORE_MASK));
    const auto stored = requestedRegion ? reader.window({requestedRegion->y, requestedRegion->x, requestedRegion->height, requestedRegion->width}) : reader.whole();
    auto points = std::make_unique<cv::Mat_<cv::Vec3f>>(int(stored.rows), int(stored.cols));
    std::memcpy(points->data, stored.xyz.data(), stored.xyz.size() * sizeof(float));
    // Invalid vertices: vc_reader's rule and the mask, and the store's (z > 0).
    for (int j = 0; j < points->rows; ++j)
        for (int i = 0; i < points->cols; ++i)
            if (!stored.valid[std::size_t(j) * std::size_t(stored.cols) + std::size_t(i)] || !vc::geometry::validSurfacePoint((*points)(j, i)))
                (*points)(j, i) = cv::Vec3f(-1.f, -1.f, -1.f);

    auto surf = std::make_unique<QuadSurface>(points.release(), scale);
    if (!requestedRegion) surf->path = path;
    surf->id   = path.filename().string();
    surf->meta = metadata;

    // Register extra channels; each loads on first use.
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

nlohmann::json bbox_to_json(const Rect3D& bbox)
{
    auto lo = nlohmann::json::array();
    lo.push_back(bbox.low[0]);
    lo.push_back(bbox.low[1]);
    lo.push_back(bbox.low[2]);
    auto hi = nlohmann::json::array();
    hi.push_back(bbox.high[0]);
    hi.push_back(bbox.high[1]);
    hi.push_back(bbox.high[2]);
    auto arr = nlohmann::json::array();
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
    cv::Mat dstToSrc;
    cv::invertAffineTransform(rotMat, dstToSrc);
    cv::Mat_<cv::Vec3f> rotatedMat(dstSize);
    const vcr::grid::Raster raster{dstToSrc.at<double>(0,2),dstToSrc.at<double>(1,2),
        dstToSrc.at<double>(0,0),dstToSrc.at<double>(1,1),dstSize.width,dstSize.height,
        dstToSrc.at<double>(0,1),dstToSrc.at<double>(1,0)};
    pointSurface()->resample(raster,rotatedMat.ptr<float>(),nullptr);
    for(auto& point : rotatedMat)
        if(!std::isfinite(cv::norm(point)))point={-1,-1,-1};

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
