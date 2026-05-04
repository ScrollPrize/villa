#include "vc/core/util/QuadSurface.hpp"

#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/LoadJson.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/util/PointIndex.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <array>
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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

// Use libtiff for BigTIFF
#include <tiffio.h>

namespace {


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

cv::Mat_<cv::Vec3f> resamplePointsLinearPreservingInvalids(
    const cv::Mat_<cv::Vec3f>& points,
    const cv::Size& newSize)
{
    cv::Mat_<cv::Vec3f> resampled(newSize.height, newSize.width,
                                  cv::Vec3f(-1.0f, -1.0f, -1.0f));
    if (points.empty()) {
        return resampled;
    }

    const float xScale = static_cast<float>(points.cols) / static_cast<float>(newSize.width);
    const float yScale = static_cast<float>(points.rows) / static_cast<float>(newSize.height);

    for (int dstY = 0; dstY < newSize.height; ++dstY) {
        float srcY = (static_cast<float>(dstY) + 0.5f) * yScale - 0.5f;
        srcY = std::clamp(srcY, 0.0f, static_cast<float>(points.rows - 1));
        const int y0 = static_cast<int>(std::floor(srcY));
        const int y1 = std::min(y0 + 1, points.rows - 1);
        const float fy = srcY - static_cast<float>(y0);

        for (int dstX = 0; dstX < newSize.width; ++dstX) {
            float srcX = (static_cast<float>(dstX) + 0.5f) * xScale - 0.5f;
            srcX = std::clamp(srcX, 0.0f, static_cast<float>(points.cols - 1));
            const int x0 = static_cast<int>(std::floor(srcX));
            const int x1 = std::min(x0 + 1, points.cols - 1);
            const float fx = srcX - static_cast<float>(x0);

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
            resampled(dstY, dstX) = top * (1.0f - fy) + bottom * fy;
        }
    }

    return resampled;
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

    constexpr double kBoundsEpsilon = 1e-5;
    for (int dstY = 0; dstY < dstSize.height; ++dstY) {
        for (int dstX = 0; dstX < dstSize.width; ++dstX) {
            double srcX = dstToSrc.at<double>(0, 0) * dstX
                        + dstToSrc.at<double>(0, 1) * dstY
                        + dstToSrc.at<double>(0, 2);
            double srcY = dstToSrc.at<double>(1, 0) * dstX
                        + dstToSrc.at<double>(1, 1) * dstY
                        + dstToSrc.at<double>(1, 2);

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
            warped(dstY, dstX) = top * (1.0f - fy) + bottom * fy;
        }
    }

    return warped;
}

} // namespace

// Axis-aligned src-sample warps used by QuadSurface::gen().  The original
// code built an affine via cv::getAffineTransform(srcf, dstf) and called
// cv::warpAffine, but the srcf/dstf construction only encodes a scale +
// translate (no rotation / shear), so the dst→src mapping is:
//     src_x = ox + dx * sx
//     src_y = oy + dy * sy
// Writing it inline buys two things the OpenCV path doesn't:
//   1. Zero per-call heap allocation.  cv::warpAffine internally allocates
//      coord lookup maps (heaptrack caught ~27 MiB × 92 calls per session
//      of churn on the render path).  These helpers only touch the pooled
//      dst buffer the caller passes in.
//   2. Straight-line hot loops that the compiler auto-vectorises on ARM
//      NEON — OpenCV's CV_32FC3 bilinear path is scalar on aarch64.
// Parallelised with OpenMP on the outer row loop, matching the rest of
// the file's existing style.
namespace {

void warpBilinearReplicateVec3f(const cv::Mat_<cv::Vec3f>& src,
                                cv::Mat_<cv::Vec3f>& dst,
                                double ox, double oy,
                                double sx, double sy)
{
    const int sc = src.cols, sr = src.rows;
    if (sc <= 0 || sr <= 0) return;
    const int dw = dst.cols, dh = dst.rows;
    const float sxmax = float(sc - 1);
    const float symax = float(sr - 1);
    const float fox = float(ox);
    const float foy = float(oy);
    const float fsx = float(sx);
    const float fsy = float(sy);
    #pragma omp parallel for schedule(dynamic, 8)
    for (int dy = 0; dy < dh; ++dy) {
        float fy = foy + float(dy) * fsy;
        fy = fy < 0.0f ? 0.0f : (fy > symax ? symax : fy);
        int y0 = int(fy);                  // floor since fy >= 0
        int y1 = y0 + 1; if (y1 > sr - 1) y1 = sr - 1;
        const float wy = fy - float(y0);
        const cv::Vec3f* row0 = src[y0];
        const cv::Vec3f* row1 = src[y1];
        cv::Vec3f* orow = dst[dy];
        for (int dx = 0; dx < dw; ++dx) {
            float fx = fox + float(dx) * fsx;
            fx = fx < 0.0f ? 0.0f : (fx > sxmax ? sxmax : fx);
            int x0 = int(fx);
            int x1 = x0 + 1; if (x1 > sc - 1) x1 = sc - 1;
            const float wx = fx - float(x0);
            const cv::Vec3f& p00 = row0[x0]; const cv::Vec3f& p01 = row0[x1];
            const cv::Vec3f& p10 = row1[x0]; const cv::Vec3f& p11 = row1[x1];
            const float iwx = 1.0f - wx;
            const float iwy = 1.0f - wy;
            orow[dx] = (p00 * iwx + p01 * wx) * iwy
                     + (p10 * iwx + p11 * wx) * wy;
        }
    }
}

void warpNearestConstU8(const cv::Mat_<uint8_t>& src,
                        cv::Mat_<uint8_t>& dst,
                        double ox, double oy,
                        double sx, double sy,
                        uint8_t border)
{
    const int sc = src.cols, sr = src.rows;
    const int dw = dst.cols, dh = dst.rows;
    const float fox = float(ox), foy = float(oy);
    const float fsx = float(sx), fsy = float(sy);
    #pragma omp parallel for schedule(dynamic, 8)
    for (int dy = 0; dy < dh; ++dy) {
        const int sy_i = int(std::lround(foy + float(dy) * fsy));
        uint8_t* orow = dst[dy];
        if (sy_i < 0 || sy_i >= sr) {
            std::memset(orow, border, size_t(dw));
            continue;
        }
        const uint8_t* srow = src[sy_i];
        for (int dx = 0; dx < dw; ++dx) {
            const int sx_i = int(std::lround(fox + float(dx) * fsx));
            orow[dx] = (sx_i < 0 || sx_i >= sc) ? border : srow[sx_i];
        }
    }
}

void warpNearestConstVec3f(const cv::Mat_<cv::Vec3f>& src,
                           cv::Mat_<cv::Vec3f>& dst,
                           double ox, double oy,
                           double sx, double sy,
                           const cv::Vec3f& border)
{
    const int sc = src.cols, sr = src.rows;
    const int dw = dst.cols, dh = dst.rows;
    const float fox = float(ox), foy = float(oy);
    const float fsx = float(sx), fsy = float(sy);
    #pragma omp parallel for schedule(dynamic, 8)
    for (int dy = 0; dy < dh; ++dy) {
        const int sy_i = int(std::lround(foy + float(dy) * fsy));
        cv::Vec3f* orow = dst[dy];
        if (sy_i < 0 || sy_i >= sr) {
            for (int dx = 0; dx < dw; ++dx) orow[dx] = border;
            continue;
        }
        const cv::Vec3f* srow = src[sy_i];
        for (int dx = 0; dx < dw; ++dx) {
            const int sx_i = int(std::lround(fox + float(dx) * fsx));
            orow[dx] = (sx_i < 0 || sx_i >= sc) ? border : srow[sx_i];
        }
    }
}

void warpBilinearConstVec3f(const cv::Mat_<cv::Vec3f>& src,
                            cv::Mat_<cv::Vec3f>& dst,
                            double ox, double oy,
                            double sx, double sy,
                            const cv::Vec3f& border)
{
    const int sc = src.cols, sr = src.rows;
    if (sc <= 0 || sr <= 0) {
        dst.setTo(border);
        return;
    }
    const int dw = dst.cols, dh = dst.rows;
    const float sxmax = float(sc - 1);
    const float symax = float(sr - 1);
    const float fox = float(ox);
    const float foy = float(oy);
    const float fsx = float(sx);
    const float fsy = float(sy);
    #pragma omp parallel for schedule(dynamic, 8)
    for (int dy = 0; dy < dh; ++dy) {
        float fy = foy + float(dy) * fsy;
        cv::Vec3f* orow = dst[dy];
        if (fy < 0.0f || fy > symax) {
            for (int dx = 0; dx < dw; ++dx) orow[dx] = border;
            continue;
        }
        int y0 = int(fy);
        int y1 = y0 + 1; if (y1 > sr - 1) y1 = sr - 1;
        const float wy = fy - float(y0);
        const cv::Vec3f* row0 = src[y0];
        const cv::Vec3f* row1 = src[y1];
        for (int dx = 0; dx < dw; ++dx) {
            float fx = fox + float(dx) * fsx;
            if (fx < 0.0f || fx > sxmax) {
                orow[dx] = border;
                continue;
            }
            int x0 = int(fx);
            int x1 = x0 + 1; if (x1 > sc - 1) x1 = sc - 1;
            const float wx = fx - float(x0);
            const cv::Vec3f& p00 = row0[x0]; const cv::Vec3f& p01 = row0[x1];
            const cv::Vec3f& p10 = row1[x0]; const cv::Vec3f& p11 = row1[x1];
            const float iwx = 1.0f - wx;
            const float iwy = 1.0f - wy;
            orow[dx] = (p00 * iwx + p01 * wx) * iwy
                     + (p10 * iwx + p11 * wx) * wy;
        }
    }
}

} // namespace

//NOTE we have 3 coordiante systems. Nominal (voxel volume) coordinates, internal relative (ptr) coords (where _center is at 0/0) and internal absolute (_points) coordinates where the upper left corner is at 0/0.
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
    _center = {static_cast<float>(_points->cols/2.0/_scale[0]), static_cast<float>(_points->rows/2.0/_scale[1]), 0.f};
}

QuadSurface::QuadSurface(cv::Mat_<cv::Vec3f> *points, const cv::Vec2f &scale)
{
    _points.reset(points);
    //-1 as many times we read with linear interpolation and access +1 locations
    _bounds = {0,0,_points->cols-1,_points->rows-1};
    _scale = scale;
    _center = {static_cast<float>(_points->cols/2.0/_scale[0]), static_cast<float>(_points->rows/2.0/_scale[1]), 0.f};
}

namespace {
static Rect3D rect_from_json(const utils::Json &json)
{
    return {{json[0][0].get_float(),json[0][1].get_float(),json[0][2].get_float()},
            {json[1][0].get_float(),json[1][1].get_float(),json[1][2].get_float()}};
}
} // anonymous namespace

QuadSurface::QuadSurface(const std::filesystem::path &path_)
{
    path = path_;
    id = path_.filename().string();
    auto metaPath = path_ / "meta.json";
    meta = vc::json::load_json_file(metaPath);

    if (meta.contains("bbox"))
        _bbox = rect_from_json(meta["bbox"]);

    if (meta.contains("components") && meta["components"].is_array()) {
        for (const auto& c : meta["components"]) {
            if (c.is_array() && c.size() >= 2)
                _components.emplace_back(c[0].get_int(), c[1].get_int());
        }
    }

    _maskTimestamp = readMaskTimestamp(path);
    _needsLoad = true;  // Points will be loaded lazily
}

QuadSurface::QuadSurface(const std::filesystem::path &path_, const utils::Json &json)
{
    path = path_;
    id = path_.filename().string();
    meta = json;

    if (json.contains("bbox"))
        _bbox = rect_from_json(json["bbox"]);

    if (json.contains("components") && json["components"].is_array()) {
        for (const auto& c : json["components"]) {
            if (c.is_array() && c.size() >= 2)
                _components.emplace_back(c[0].get_int(), c[1].get_int());
        }
    }

    _maskTimestamp = readMaskTimestamp(path);
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

    trimToValidBbox();

    // Keep existing bbox and meta if already set, otherwise take from loaded
    if (_bbox.low[0] == 0 && _bbox.high[0] == 0) {
        _bbox = loaded->_bbox;
    }

    _maskTimestamp = readMaskTimestamp(path);

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

    cv::Rect bounds = {0,0,_points->cols-2,_points->rows-2};
    if (!bounds.contains(cv::Point(p[0],p[1])))
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

cv::Vec2f QuadSurface::scale() const
{
    return _scale;
}

cv::Vec3f QuadSurface::center() const
{
    return _center;
}

cv::Vec2f QuadSurface::ptrToGrid(const cv::Vec3f& ptr) const
{
    return {ptr[0] + _center[0] * _scale[0],
            ptr[1] + _center[1] * _scale[1]};
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
        if (channel.empty()) {
            // On-demand loading
            std::filesystem::path channel_path = path / (name + ".tif");
            if (std::filesystem::exists(channel_path)) {
                std::vector<cv::Mat> layers;
                cv::imreadmulti(channel_path.string(), layers, cv::IMREAD_UNCHANGED);
                if (!layers.empty()) {
                    channel = layers[0];
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

bool QuadSurface::trimToValidBbox()
{
    if (!_points || _points->empty()) return false;
    const int rows = _points->rows;
    const int cols = _points->cols;
    int r0 = rows, r1 = -1, c0 = cols, c1 = -1;
    for (int r = 0; r < rows; r++) {
        const cv::Vec3f* row = (*_points)[r];
        int rc0 = cols, rc1 = -1;
        for (int c = 0; c < cols; c++) {
            if (row[c][0] != -1.f) {
                if (c < rc0) rc0 = c;
                rc1 = c;
            }
        }
        if (rc1 >= 0) {
            if (r < r0) r0 = r;
            r1 = r;
            if (rc0 < c0) c0 = rc0;
            if (rc1 > c1) c1 = rc1;
        }
    }
    if (r1 < 0 || c1 < 0) return false;
    const int bbH = r1 - r0 + 1;
    const int bbW = c1 - c0 + 1;
    // Skip only if nothing to trim (bbox already matches grid exactly).
    if (bbH == rows && bbW == cols) return false;
    const std::size_t origBytes = std::size_t(rows) * cols * sizeof(cv::Vec3f);
    const std::size_t trimBytes = std::size_t(bbH) * bbW * sizeof(cv::Vec3f);
    const double pctSaved = 1.0 - double(trimBytes) / double(origBytes);
    auto trimmed = std::make_unique<cv::Mat_<cv::Vec3f>>(
        (*_points)(cv::Rect(c0, r0, bbW, bbH)).clone());
    _points = std::move(trimmed);
    // Shift center: after cropping by (c0, r0), the new grid index 0
    // corresponds to the old index (c0, r0), so we must subtract.
    _center[0] -= float(c0) / _scale[0];
    _center[1] -= float(r0) / _scale[1];
    _bounds = {0, 0, bbW - 1, bbH - 1};
    _bbox = {{-1, -1, -1}, {-1, -1, -1}};  // World bbox cached, needs recompute.
    _validMaskCache = cv::Mat_<uint8_t>();
    _validMaskAllValid = false;
    _normalCache = cv::Mat_<cv::Vec3f>();
    if (DebugLoggingEnabled()) {
        std::fprintf(stderr,
            "[SURF] trim %s %dx%d -> %dx%d  saved %zu MB (%.1f%%)\n",
            id.c_str(), cols, rows, bbW, bbH,
            (origBytes - trimBytes) / (1024 * 1024),
            pctSaved * 100.0);
    }
    return true;
}

void QuadSurface::unloadPoints()
{
    if (path.empty()) return;  // No disk backing — can't reload.
    std::lock_guard<std::mutex> lock(_loadMutex);
    if (_needsLoad) return;    // Already unloaded.
    std::size_t mb = 0;
    if (_points) {
        mb = static_cast<std::size_t>(_points->rows) * _points->cols
             * sizeof(cv::Vec3f) / (1024 * 1024);
    }
    _points.reset();
    _channels.clear();
    _validMaskCache = cv::Mat_<uint8_t>();
    _validMaskAllValid = false;
    _normalCache = cv::Mat_<cv::Vec3f>();
    _needsLoad = true;
    if (DebugLoggingEnabled()) {
        std::fprintf(stderr, "[SURF] unload %s (%zu MB freed)\n", id.c_str(), mb);
    }
}

void QuadSurface::unloadCaches()
{
    _validMaskCache = cv::Mat_<uint8_t>();
    _validMaskAllValid = false;
    _normalCache = cv::Mat_<cv::Vec3f>();
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

    if (!_validMaskCache.empty() &&
        _validMaskCache.rows == _points->rows &&
        _validMaskCache.cols == _points->cols) {
        return _validMaskCache;
    }

    const int rows = _points->rows;
    const int cols = _points->cols;
    cv::Mat_<uint8_t> mask(rows, cols);

    // Tight per-row loop:
    //  - NaN check via self-inequality (NaN != NaN) — no math.h call, branchless.
    //  - Sentinel check is a contiguous-byte equality against (-1, -1, -1).
    //  - Bitwise & / | so the compiler emits straight-line NEON, no early exit.
    // Inf is treated as valid here (was rejected by std::isfinite); coordinates
    // never go to ±Inf in the surface pipeline, so this is a safe relaxation.
    // Per-row "any invalid" accumulator for the all-valid fast path in
    // gen() below. OR-reduced across threads after the loop.
    std::vector<uint8_t> anyInvalidPerRow(rows, 0);
#pragma omp parallel for schedule(dynamic, 16)
    for (int j = 0; j < rows; ++j) {
        const cv::Vec3f* __restrict__ row = (*_points)[j];
        uint8_t* __restrict__ dst = mask[j];
        uint8_t rowAnyInvalid = 0;
        for (int i = 0; i < cols; ++i) {
            const float a = row[i][0], b = row[i][1], c = row[i][2];
            const int nan = (a != a) | (b != b) | (c != c);
            const int sent = (a == -1.f) & (b == -1.f) & (c == -1.f);
            const uint8_t invalid = (nan | sent) ? uint8_t(1) : uint8_t(0);
            dst[i] = invalid ? uint8_t(0) : uint8_t(255);
            rowAnyInvalid |= invalid;
        }
        anyInvalidPerRow[j] = rowAnyInvalid;
    }
    uint8_t anyInvalid = 0;
    for (uint8_t v : anyInvalidPerRow) anyInvalid |= v;
    _validMaskAllValid = (anyInvalid == 0);
    _validMaskCache = mask;
    return mask;
}

void QuadSurface::writeValidMask(const cv::Mat& img)
{
    if (path.empty()) {
        return;
    }
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
        _center = {
            static_cast<float>(_points->cols / 2.0 / _scale[0]),
            static_cast<float>(_points->rows / 2.0 / _scale[1]),
            0.f
        };
    } else {
        _bounds = {0, 0, -1, -1};
        _center = {0, 0, 0};
    }

    _bbox = {{-1, -1, -1}, {-1, -1, -1}};
    _validMaskCache = cv::Mat_<uint8_t>();
    _validMaskAllValid = false;
}

void QuadSurface::gen(cv::Mat_<cv::Vec3f>* coords,
                      cv::Mat_<cv::Vec3f>* normals,
                      cv::Size size,
                      const cv::Vec3f& ptr,
                      float scale,
                      const cv::Vec3f& offset) const
{
    const_cast<QuadSurface*>(this)->ensureLoaded();
    const bool need_normals = (normals != nullptr) || offset[2] || ptr[2];

    const cv::Vec3f ul = internal_loc(offset/scale + _center, ptr, _scale);
    const int w = size.width, h = size.height;

    cv::Mat_<cv::Vec3f> coords_local, normals_local;
    if (!coords)  coords  = &coords_local;
    if (!normals) normals = &normals_local;

    coords->create(size + cv::Size(8, 8));

    // --- build mapping  ---------------------------------
    // Axis-aligned scale+translate; see the bespoke warp helpers above.
    const double sx = static_cast<double>(_scale[0]) / static_cast<double>(scale);
    const double sy = static_cast<double>(_scale[1]) / static_cast<double>(scale);
    const double ox = static_cast<double>(ul[0]) - 4.0 * sx;
    const double oy = static_cast<double>(ul[1]) - 4.0 * sy;

    // --- build a source validity mask (255 if point is valid) -------------
    // Trigger the cache build + set _validMaskAllValid before deciding
    // whether we need the validity warp below.
    cv::Mat_<uint8_t> valid_src = validMask();
    bool skipValidity = _validMaskAllValid;

    // --- warp coords and validity ----------------------------------------
    cv::Mat_<cv::Vec3f>& coords_big = _genCoordsScratch;
    cv::Mat_<uint8_t>& valid_big = _genValidScratch;

    if (!_components.empty()) {
        // Multi-component surface: warp each component separately with
        // constant NaN border so no interpolation across component boundaries.
        const cv::Vec3f nanV(std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN());
        coords_big.create(h + 8, w + 8);
        coords_big.setTo(nanV);
        valid_big.create(h + 8, w + 8);
        valid_big.setTo(0);

        const int rows = _points->rows;
        cv::Mat_<cv::Vec3f> compCoords;
        cv::Mat_<uint8_t> compValidBig;
        for (const auto& [c0, c1] : _components) {
            const int cw = c1 - c0;
            if (cw <= 0 || c0 < 0 || c1 > _points->cols) continue;

            cv::Mat_<cv::Vec3f> compPts = (*_points)(cv::Rect(c0, 0, cw, rows));
            compCoords.create(h + 8, w + 8);
            warpBilinearConstVec3f(compPts, compCoords, ox - c0, oy, sx, sy, nanV);

            if (!skipValidity) {
                cv::Mat_<uint8_t> compValid = valid_src(cv::Rect(c0, 0, cw, rows));
                compValidBig.create(h + 8, w + 8);
                warpNearestConstU8(compValid, compValidBig, ox - c0, oy, sx, sy, 0);
                compCoords.copyTo(coords_big, compValidBig);
                cv::bitwise_or(valid_big, compValidBig, valid_big);
            } else {
                // All-valid source: composite non-NaN pixels directly
                for (int r = 0; r < coords_big.rows; r++) {
                    const cv::Vec3f* s = compCoords[r];
                    cv::Vec3f* d = coords_big[r];
                    for (int ci = 0; ci < coords_big.cols; ci++) {
                        if (std::isfinite(s[ci][0])) d[ci] = s[ci];
                    }
                }
            }
        }
    } else {
        // Single component: replicate coords, constant-0 validity.
        // Always warp validity even when all source points are valid —
        // the 4px halo around the crop region needs 0s so the
        // invalidation pass below sets them to NaN (black edges).
        coords_big.create(h + 8, w + 8);
        warpBilinearReplicateVec3f(*_points, coords_big, ox, oy, sx, sy);
        valid_big.create(h + 8, w + 8);
        warpNearestConstU8(valid_src, valid_big, ox, oy, sx, sy, 0);
        skipValidity = false;  // force invalidation pass below
    }

    // --- normals: warp cached source-grid normals -------------------
    cv::Mat_<cv::Vec3f>& normals_big = _genNormalsScratch;
    if (need_normals) {
        // Build source-grid normal cache once per surface. Subsequent gen()
        // calls (panning, zooming) reuse it. Cleared by unloadCaches() when
        // a different surface becomes active.
        if (_normalCache.empty() || _normalCache.size() != _points->size()) {
            _normalCache.create(_points->rows, _points->cols);
            const cv::Vec3f qn(std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::quiet_NaN());
            const int rows = _points->rows;
            const int cols = _points->cols;
            // Border rows/cols: no ±1 neighbors, fill with NaN sentinel.
            if (rows > 0) {
                cv::Vec3f* top = _normalCache[0];
                cv::Vec3f* bot = _normalCache[rows - 1];
                for (int c = 0; c < cols; c++) { top[c] = qn; bot[c] = qn; }
            }
            #pragma omp parallel for schedule(dynamic, 16)
            for (int r = 1; r < rows - 1; r++) {
                cv::Vec3f* dst = _normalCache[r];
                const cv::Vec3f* row = (*_points)[r];
                dst[0] = qn;
                dst[cols - 1] = qn;
                for (int c = 1; c < cols - 1; c++) {
                    if (row[c][0] == -1.f) {
                        dst[c] = qn;
                    } else {
                        dst[c] = grid_normal_int(*_points, r, c);
                    }
                }
            }
        }
        const cv::Vec3f qnVec(std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::quiet_NaN());
        normals_big.create(h + 8, w + 8);
        warpNearestConstVec3f(_normalCache, normals_big,
                              ox, oy, sx, sy, qnVec);
    }

    // --- crop away the 4px halo ----------------------------------------
    // Take views, not clones: ref-counted buffers stay alive via the shared
    // Mat header. Skips ~48MB/frame of memcpy for a 1920x1080 composite.
    cv::Rect inner(4, 4, w, h);
    *coords = coords_big(inner);
    if (need_normals) {
        *normals = normals_big(inner);
    }

    // --- invalidate out-of-footprint pixels (kill GUI leakage) ----------
    // When the source mask is all-valid the cropped image has no 0s, so
    // skip the invalidation pass entirely. Otherwise walk coords + normals
    // once together (fused vs OpenCV's valid==0 + setTo pair of passes).
    const cv::Vec3f qnan(std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN());
    if (!skipValidity) {
        cv::Mat valid = valid_big(inner);
        const bool doNormals = need_normals;
        for (int j = 0; j < h; ++j) {
            const uint8_t* __restrict__ v = valid.ptr<uint8_t>(j);
            cv::Vec3f* __restrict__ crow = coords->ptr<cv::Vec3f>(j);
            cv::Vec3f* __restrict__ nrow = doNormals
                ? normals->ptr<cv::Vec3f>(j) : nullptr;
            for (int i = 0; i < w; ++i) {
                if (!v[i]) {
                    crow[i] = qnan;
                    if (doNormals) nrow[i] = qnan;
                }
            }
        }
    }

    if (need_normals && ul[2] != 0.0f) {
        const float off = ul[2];
        for (int j = 0; j < h; ++j) {
            const cv::Vec3f* nrow = normals->ptr<cv::Vec3f>(j);
            cv::Vec3f* crow = coords->ptr<cv::Vec3f>(j);
            for (int i = 0; i < w; ++i) {
                const cv::Vec3f& n = nrow[i];
                if (!std::isnan(n[0]) && !std::isnan(n[1]) && !std::isnan(n[2])) {
                    crow[i][0] += n[0] * off;
                    crow[i][1] += n[1] * off;
                    crow[i][2] += n[2] * off;
                }
            }
        }
    }
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

template <typename E>
static float search_min_loc(const cv::Mat_<E> &points, cv::Vec2f &loc, cv::Vec3f &out, cv::Vec3f tgt, cv::Vec2f init_step, float min_step_x)
{
    cv::Rect boundary(1,1,points.cols-2,points.rows-2);
    if (!boundary.contains(cv::Point(loc))) {
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
            if (!boundary.contains(cv::Point(cand)))
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
    loc = cv::Vec2f(points.cols/2,points.rows/2);
    cv::Vec3f _out;

    cv::Vec2f step_small = {std::max(1.0f,scale),std::max(1.0f,scale)};
    float min_mul = std::min(0.1*points.cols/scale,0.1*points.rows/scale);
    cv::Vec2f step_large = {min_mul*scale,min_mul*scale};

    assert(points.cols > 3);
    assert(points.rows > 3);

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
        loc = {static_cast<float>(1 + (rand() % (points.cols-3))), static_cast<float>(1 + (rand() % (points.rows-3)))};

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
                           SurfacePatchIndex* surfaceIndex, PointIndex* pointIndex)
{
    ensureLoaded();
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

    // Try point index for a better starting hint
    if (pointIndex && !pointIndex->empty()) {
        auto nearest = pointIndex->nearest(tgt, th * 4.0f);
        if (nearest) {
            // Use nearest point as starting location hint
            cv::Vec3f hint_ptr{0, 0, 0};
            // Recursively call without indices to avoid infinite loop
            float hint_dist = pointTo(hint_ptr, nearest->position, th, std::max(1, max_iters / 4), nullptr, nullptr);
            if (hint_dist >= 0 && hint_dist < min_dist) {
                // Now search from this hint toward our actual target
                loc = cv::Vec2f(hint_ptr[0], hint_ptr[1]) + cv::Vec2f(_center[0]*_scale[0], _center[1]*_scale[1]);
                dist = search_min_loc(*_points, loc, _out, tgt, step_small, _scale[0]*0.1);
                if (dist < th && dist >= 0) {
                    ptr = cv::Vec3f(loc[0], loc[1], 0) - cv::Vec3f(_center[0]*_scale[0], _center[1]*_scale[1], 0);
                    return dist;
                } else if (dist >= 0 && dist < min_dist) {
                    min_loc = loc;
                    min_dist = dist;
                }
            }
        }
    }

    // Fall back to random sampling if indices didn't help
    int r_full = 0;
    int skip_count = 0;
    for(int r=0; r<10*max_iters && r_full<max_iters; r++) {
        loc = {static_cast<float>(1 + (rand() % (_points->cols-3))), static_cast<float>(1 + (rand() % (_points->rows-3)))};

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

void QuadSurface::save(const std::filesystem::path &path_, bool force_overwrite)
{
    if (path_.filename().empty())
        save(path_, path_.parent_path().filename(), force_overwrite);
    else
        save(path_, path_.filename(), force_overwrite);
}

void QuadSurface::saveOverwrite()
{
    if (path.empty()) {
        throw std::runtime_error("QuadSurface::saveOverwrite() requires a valid path");
    }

    std::filesystem::path final_path = path;
    std::string uuid = !id.empty() ? id : final_path.filename().string();
    if (uuid.empty()) {
        throw std::runtime_error("QuadSurface::saveOverwrite() requires a non-empty uuid");
    }

    save(final_path.string(), uuid, true);
    path = final_path;
    id = uuid;
}

void QuadSurface::invalidateMask()
{
    // Clear from memory
    _channels.erase("mask");
    _validMaskCache = cv::Mat_<uint8_t>();
    _validMaskAllValid = false;

    // Delete from disk
    if (!path.empty()) {
        std::filesystem::path maskFile = path / "mask.tif";
        std::error_code ec;
        std::filesystem::remove(maskFile, ec);
    }
}

void QuadSurface::writeDataToDirectory(const std::filesystem::path& dir, const std::string& skipChannel)
{
    // Trim padding before serializing so the on-disk grid matches the in-RAM
    // one. No-op if already trimmed or saving would save <25%.
    trimToValidBbox();

    // Split the points matrix into x, y, z channels
    std::vector<cv::Mat> xyz;
    cv::split((*_points), xyz);

    // QuadSurface tifxyz files are written as untiled, uncompressed TIFFs.
    writeTiff(dir / "x.tif", xyz[0], -1, 0, 0, -1.0f, COMPRESSION_NONE, dpi_);
    writeTiff(dir / "y.tif", xyz[1], -1, 0, 0, -1.0f, COMPRESSION_NONE, dpi_);
    writeTiff(dir / "z.tif", xyz[2], -1, 0, 0, -1.0f, COMPRESSION_NONE, dpi_);

    // OpenCV compression params for fallback
    std::vector<int> compression_params = { cv::IMWRITE_TIFF_COMPRESSION, COMPRESSION_NONE };

    // Save additional channels
    for (auto const& [name, mat] : _channels) {
        if (!mat.empty() && (skipChannel.empty() || name != skipChannel)) {
            bool wrote = false;

            // Try untiled, uncompressed TIFF for single-channel ancillary data (8U/16U/32F)
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

            // Fallback to OpenCV for multi-channel or other formats
            if (!wrote) {
                cv::imwrite((dir / (name + ".tif")).string(), mat, compression_params);
            }
        }
    }
}

void QuadSurface::saveSnapshot(int maxBackups)
{
    if (path.empty()) {
        throw std::runtime_error("QuadSurface::saveSnapshot() requires a valid path");
    }

    // Path is expected to be: /path/to/scroll.volpkg/paths/segment_name
    std::filesystem::path volpkgRoot = path.parent_path().parent_path();
    std::string segmentName = path.filename().string();

    // Create centralized backups directory structure
    std::filesystem::path backupsDir = volpkgRoot / "backups" / segmentName;

    std::error_code ec;
    std::filesystem::create_directories(backupsDir, ec);
    if (ec) {
        throw std::runtime_error("Failed to create backups directory: " + ec.message());
    }

    // Find existing backup directories and determine next backup number
    std::vector<int> existingBackups;
    if (std::filesystem::exists(backupsDir)) {
        for (const auto& entry : std::filesystem::directory_iterator(backupsDir)) {
            if (entry.is_directory()) {
                try {
                    int backupNum = std::stoi(entry.path().filename().string());
                    if (backupNum >= 0 && backupNum < maxBackups) {
                        existingBackups.push_back(backupNum);
                    }
                } catch (...) {
                    // Skip non-numeric directories
                }
            }
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

    // Write surface data (skip mask - we'll copy it from disk instead)
    writeDataToDirectory(snapshot_dest, "mask");

    // Write metadata - create a copy so we don't modify the original
    utils::Json snapshotMeta = meta;
    {
        auto lo = utils::Json::array();
        lo.push_back(bbox().low[0]); lo.push_back(bbox().low[1]); lo.push_back(bbox().low[2]);
        auto hi = utils::Json::array();
        hi.push_back(bbox().high[0]); hi.push_back(bbox().high[1]); hi.push_back(bbox().high[2]);
        auto bb = utils::Json::array();
        bb.push_back(std::move(lo)); bb.push_back(std::move(hi));
        snapshotMeta["bbox"] = std::move(bb);
    }
    snapshotMeta["type"] = "seg";
    snapshotMeta["uuid"] = id;
    snapshotMeta["format"] = "tifxyz";
    {
        auto sc = utils::Json::array();
        sc.push_back(_scale[0]); sc.push_back(_scale[1]);
        snapshotMeta["scale"] = std::move(sc);
    }

    std::ofstream o(snapshot_dest / "meta.json");
    o << snapshotMeta.dump(4) << std::endl;
    o.close();

    // Copy mask.tif and generations.tif if they exist on disk
    std::filesystem::path maskFile = path / "mask.tif";
    std::filesystem::path generationsFile = path / "generations.tif";

    if (std::filesystem::exists(maskFile)) {
        std::filesystem::path destMask = snapshot_dest / "mask.tif";
        std::filesystem::copy_file(maskFile, destMask,
            std::filesystem::copy_options::overwrite_existing, ec);
    }

    if (std::filesystem::exists(generationsFile)) {
        std::filesystem::path destGenerations = snapshot_dest / "generations.tif";
        std::filesystem::copy_file(generationsFile, destGenerations,
            std::filesystem::copy_options::overwrite_existing, ec);
    }
}


void QuadSurface::save(const std::string &path_, const std::string &uuid, bool force_overwrite)
{
    std::filesystem::path target_path = path_;
    std::filesystem::path final_path = path_;

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

    // Preserve auxiliary files from the existing directory before replacing
    if (force_overwrite && std::filesystem::exists(final_path)) {
        // Copy corrections.json if it exists
        std::filesystem::path correctionsFile = final_path / "corrections.json";
        if (std::filesystem::exists(correctionsFile)) {
            std::error_code copyEc;
            std::filesystem::copy_file(correctionsFile, temp_path / "corrections.json",
                std::filesystem::copy_options::overwrite_existing, copyEc);
            // Ignore errors - corrections are not critical for surface integrity
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
    id = uuid;
}

void QuadSurface::save_meta()
{
    if (meta.is_null())
        throw std::runtime_error("can't save_meta() without metadata!");
    if (path.empty())
        throw std::runtime_error("no storage path for QuadSurface");

    if (!meta.is_object()) {
        throw std::runtime_error("can't save_meta() with non-object metadata!");
    }

    const std::string uuid = !id.empty() ? id : path.filename().string();
    if (uuid.empty()) {
        throw std::runtime_error("QuadSurface::save_meta() requires a non-empty uuid");
    }

    {
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

    std::ofstream o(path/"meta.json.tmp");
    o << meta.dump(4) << std::endl;

    //rename to make creation atomic
    std::filesystem::rename(path/"meta.json.tmp", path/"meta.json");
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

std::unique_ptr<QuadSurface> load_quad_from_tifxyz(const std::string &path, int flags)
{
    auto read_band_into = [](const std::filesystem::path& fpath,
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

        // Allocate destination if first band
        if (points.empty()) {
            points.create(static_cast<int>(H), static_cast<int>(W));
            // Initialize as invalid
            for (int y=0;y<points.rows;++y)
                for (int x=0;x<points.cols;++x)
                    points(y,x) = cv::Vec3f(-1.f,-1.f,-1.f);
            outW = static_cast<int>(W);
            outH = static_cast<int>(H);
        } else {
            if (outW != static_cast<int>(W) || outH != static_cast<int>(H)) {
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

            for (uint32_t y0=0; y0<H; y0+=tileH) {
                const uint32_t dy = std::min(tileH, H - y0);
                for (uint32_t x0=0; x0<W; x0+=tileW) {
                    const uint32_t dx = std::min(tileW, W - x0);
                    const ttile_t tidx = TIFFComputeTile(tif, x0, y0, 0, 0);
                    tmsize_t n = TIFFReadEncodedTile(tif, tidx, tileBuf.data(), tileBytes);
                    if (n < 0) {
                        // fill with zeros on read error
                        std::fill(tileBuf.begin(), tileBuf.end(), 0);
                    }
                    for (uint32_t ty=0; ty<dy; ++ty) {
                        const uint8_t* row = tileBuf.data() + (static_cast<size_t>(ty)*tileW*bytesPer);
                        for (uint32_t tx=0; tx<dx; ++tx) {
                            float fv = to_float(row + static_cast<size_t>(tx)*bytesPer);
                            cv::Vec3f& dst = points(static_cast<int>(y0+ty), static_cast<int>(x0+tx));
                            dst[channel] = fv;
                        }
                    }
                }
            }
        } else {
            const tmsize_t scanBytes = TIFFScanlineSize(tif);
            std::vector<uint8_t> scanBuf(static_cast<size_t>(scanBytes));
            for (uint32_t y=0; y<H; ++y) {
                if (TIFFReadScanline(tif, scanBuf.data(), y, 0) != 1) {
                    std::fill(scanBuf.begin(), scanBuf.end(), 0);
                }
                const uint8_t* row = scanBuf.data();
                for (uint32_t x=0; x<W; ++x) {
                    float fv = to_float(row + static_cast<size_t>(x)*bytesPer);
                    cv::Vec3f& dst = points(static_cast<int>(y), static_cast<int>(x));
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

    // Invalidate by z<=0
    for (int j=0;j<points->rows;++j)
        for (int i=0;i<points->cols;++i)
            if ((*points)(j,i)[2] <= 0.f)
                (*points)(j,i) = cv::Vec3f(-1.f,-1.f,-1.f);

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
                        if (dstX < static_cast<uint32_t>(W) && dstY < static_cast<uint32_t>(H)) {
                            (*points)(static_cast<int>(dstY), static_cast<int>(dstX)) = invalidPoint;
                        }
                    };
                    if (TIFFIsTiled(mtif)) {
                        uint32_t tileW=0, tileH=0;
                        TIFFGetField(mtif, TIFFTAG_TILEWIDTH,  &tileW);
                        TIFFGetField(mtif, TIFFTAG_TILELENGTH, &tileH);
                        const tmsize_t tileBytes = TIFFTileSize(mtif);
                        std::vector<uint8_t> tileBuf(static_cast<size_t>(tileBytes));
                        for (uint32_t y0=0; y0<mH; y0+=tileH) {
                            const uint32_t dy = std::min(tileH, mH - y0);
                            for (uint32_t x0=0; x0<mW; x0+=tileW) {
                                const uint32_t dx = std::min(tileW, mW - x0);
                                const ttile_t tidx = TIFFComputeTile(mtif, x0, y0, 0, 0);
                                tmsize_t n = TIFFReadEncodedTile(mtif, tidx, tileBuf.data(), tileBytes);
                                if (n < 0) std::fill(tileBuf.begin(), tileBuf.end(), 0);
                                const size_t tileRowStride = static_cast<size_t>(tileW) * pixelStride;
                                for (uint32_t ty=0; ty<dy; ++ty) {
                                    const uint8_t* row = tileBuf.data() + static_cast<size_t>(ty) * tileRowStride;
                                    for (uint32_t tx=0; tx<dx; ++tx) {
                                        const uint8_t* px = row + static_cast<size_t>(tx) * pixelStride;
                                        if (!to_valid(px)) {
                                            invalidate(x0 + tx, y0 + ty);
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        const tmsize_t scanBytes = TIFFScanlineSize(mtif);
                        std::vector<uint8_t> scanBuf(static_cast<size_t>(scanBytes));
                        for (uint32_t y=0; y<mH; ++y) {
                            if (TIFFReadScanline(mtif, scanBuf.data(), y, 0) != 1) {
                                std::fill(scanBuf.begin(), scanBuf.end(), 0);
                            }
                            const uint8_t* row = scanBuf.data();
                            for (uint32_t x=0; x<mW; ++x) {
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
    surf->path = path;
    surf->id   = metadata["uuid"].get_string();
    surf->meta = metadata;

    // Register extra channels lazily (left as OpenCV-based on-demand load).
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (entry.path().extension() == ".tif") {
            std::string filename = entry.path().stem().string();
            if (filename != "x" && filename != "y" && filename != "z") {
                surf->setChannel(filename, cv::Mat());
            }
        }
    }
    return surf;
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

std::unique_ptr<QuadSurface> surface_diff(QuadSurface* a, QuadSurface* b, float tolerance) {
    auto diff_points = std::make_unique<cv::Mat_<cv::Vec3f>>(a->rawPoints().clone());

    int width = diff_points->cols;
    int height = diff_points->rows;

    if (!intersect(a->bbox(), b->bbox())) {
        return std::make_unique<QuadSurface>(diff_points.release(), a->scale());
    }

    // Build spatial index for surface b
    PointIndex bIndex;
    bIndex.buildFromMat(b->rawPoints());

    int removed_count = 0;
    int total_valid = 0;
    const float toleranceSq = tolerance * tolerance;

    #pragma omp parallel for reduction(+:removed_count,total_valid)
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            cv::Vec3f point = (*diff_points)(j, i);

            if (point[0] == -1.f) {
                continue;
            }

            total_valid++;

            auto result = bIndex.nearest(point, tolerance);
            if (result && result->distanceSq <= toleranceSq) {
                (*diff_points)(j, i) = {-1, -1, -1};
                removed_count++;
            }
        }
    }

    std::cout << "Surface diff: removed " << removed_count
              << " points out of " << total_valid << " valid points" << std::endl;

    return std::make_unique<QuadSurface>(diff_points.release(), a->scale());
}

std::unique_ptr<QuadSurface> surface_union(QuadSurface* a, QuadSurface* b, float tolerance) {
    auto union_points = std::make_unique<cv::Mat_<cv::Vec3f>>(a->rawPoints().clone());

    // Build spatial index for surface a
    PointIndex aIndex;
    aIndex.buildFromMat(*union_points);

    const cv::Mat_<cv::Vec3f>& b_points = b->rawPoints();
    const float toleranceSq = tolerance * tolerance;

    int added_count = 0;

    // Add points from b that don't exist in a
    for (int j = 0; j < b_points.rows; j++) {
        for (int i = 0; i < b_points.cols; i++) {
            const cv::Vec3f& point_b = b_points(j, i);

            if (point_b[0] == -1.f) {
                continue;
            }

            // Check if this point exists in a
            auto result = aIndex.nearest(point_b, tolerance);

            // If point is not found in a, we need to add it
            if (!result || result->distanceSq > toleranceSq) {
                int grid_x = std::round(i * b->scale()[0] / a->scale()[0]);
                int grid_y = std::round(j * b->scale()[1] / a->scale()[1]);

                if (grid_x >= 0 && grid_x < union_points->cols &&
                    grid_y >= 0 && grid_y < union_points->rows) {

                    if ((*union_points)(grid_y, grid_x)[0] == -1.f) {
                        (*union_points)(grid_y, grid_x) = point_b;
                        added_count++;
                    }
                }
            }
        }
    }

    std::cout << "Surface union: added " << added_count << " points from surface b" << std::endl;

    return std::make_unique<QuadSurface>(union_points.release(), a->scale());
}

std::unique_ptr<QuadSurface> surface_intersection(QuadSurface* a, QuadSurface* b, float tolerance) {
    auto intersect_points = std::make_unique<cv::Mat_<cv::Vec3f>>(a->rawPoints().clone());

    int width = intersect_points->cols;
    int height = intersect_points->rows;

    // Build spatial index for surface b
    PointIndex bIndex;
    bIndex.buildFromMat(b->rawPoints());

    int kept_count = 0;
    int total_valid = 0;
    const float toleranceSq = tolerance * tolerance;

    // Keep only points that exist in both surfaces
    #pragma omp parallel for reduction(+:kept_count,total_valid)
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            cv::Vec3f point_a = (*intersect_points)(j, i);

            if (point_a[0] == -1.f) {
                continue;
            }

            total_valid++;

            auto result = bIndex.nearest(point_a, tolerance);
            if (result && result->distanceSq <= toleranceSq) {
                // Point exists in both - keep it
                kept_count++;
            } else {
                // Point doesn't exist in b - remove it
                (*intersect_points)(j, i) = {-1, -1, -1};
            }
        }
    }

    std::cout << "Surface intersection: kept " << kept_count
              << " points out of " << total_valid << " valid points" << std::endl;

    return std::make_unique<QuadSurface>(intersect_points.release(), a->scale());
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
    _bbox = {{-1, -1, -1}, {-1, -1, -1}};
}

void QuadSurface::resample(float factor, int interpolation)
{
    resample(factor, factor, interpolation);
}

void QuadSurface::resample(float factor_x, float factor_y, int interpolation)
{
    ensureLoaded();
    if (!_points || _points->empty()) {
        return;
    }
    if (factor_x <= 0.0f || factor_y <= 0.0f) {
        return;
    }
    if (std::abs(factor_x - 1.0f) < 0.001f && std::abs(factor_y - 1.0f) < 0.001f) {
        return;
    }

    // Calculate new size
    cv::Size newSize(
        std::max(1, static_cast<int>(std::round(_points->cols * factor_x))),
        std::max(1, static_cast<int>(std::round(_points->rows * factor_y)))
    );

    // Resample points without blending across invalid markers.
    cv::Mat_<cv::Vec3f> resampledMat;
    if (interpolation == cv::INTER_LINEAR) {
        resampledMat = resamplePointsLinearPreservingInvalids(*_points, newSize);
    } else {
        cv::resize(*_points, resampledMat, newSize, 0, 0, interpolation);
    }

    // Update points
    *_points = resampledMat;

    // Resample all channels
    for (auto& [name, channel] : _channels) {
        if (channel.empty()) continue;

        cv::Mat resampledChannel;
        // Use nearest neighbor for mask-like channels, linear for others
        int chanInterp = (name == "mask" || name == "generations")
                         ? cv::INTER_NEAREST : interpolation;
        cv::resize(channel, resampledChannel, newSize, 0, 0, chanInterp);
        channel = resampledChannel;
    }

    // Update scale to maintain physical size
    _scale[0] /= factor_x;
    _scale[1] /= factor_y;

    // Invalidate cached bbox
    _bbox = {{-1, -1, -1}, {-1, -1, -1}};
}

float QuadSurface::computeZOrientationAngle() const
{
    const_cast<QuadSurface*>(this)->ensureLoaded();
    if (!_points || _points->empty()) {
        return 0.0f;
    }

    // Find min/max Z to build a stable weighting for the high-Z centroid.
    double minZ = std::numeric_limits<double>::max();
    double maxZ = std::numeric_limits<double>::lowest();
    for (int row = 0; row < _points->rows; ++row) {
        for (int col = 0; col < _points->cols; ++col) {
            const cv::Vec3f& pt = (*_points)(row, col);
            if (pt[0] == -1.f || !std::isfinite(pt[2])) {
                continue;
            }
            minZ = std::min(minZ, static_cast<double>(pt[2]));
            maxZ = std::max(maxZ, static_cast<double>(pt[2]));
        }
    }

    if (!(minZ < maxZ)) {
        return 0.0f;
    }

    // Compute a Z-weighted centroid so the orientation follows the dominant Z gradient.
    double sumW = 0.0;
    double sumRow = 0.0;
    double sumCol = 0.0;
    for (int row = 0; row < _points->rows; ++row) {
        for (int col = 0; col < _points->cols; ++col) {
            const cv::Vec3f& pt = (*_points)(row, col);
            if (pt[0] == -1.f || !std::isfinite(pt[2])) {
                continue;
            }
            const double w = static_cast<double>(pt[2]) - minZ;
            if (w <= 0.0) {
                continue;
            }
            sumW += w;
            sumRow += static_cast<double>(row) * w;
            sumCol += static_cast<double>(col) * w;
        }
    }

    if (sumW <= 0.0) {
        return 0.0f;
    }

    const double centroidRow = sumRow / sumW;
    const double centroidCol = sumCol / sumW;

    // Grid center (OpenCV: x = col, y = row)
    const double centerRow = (_points->rows - 1) / 2.0;
    const double centerCol = (_points->cols - 1) / 2.0;

    // Vector from center to Z-weighted centroid in OpenCV coordinates.
    const double dx = centroidCol - centerCol;  // x = col (right)
    const double dy = centroidRow - centerRow;  // y = row (down)

    // If centroid is at center, no rotation needed.
    if (std::abs(dx) < 1e-6 && std::abs(dy) < 1e-6) {
        return 0.0f;
    }

    // Angle from +X (right) in OpenCV image coordinates.
    const double angleRad = std::atan2(dy, dx);
    const double angleDeg = angleRad * 180.0 / M_PI;

    // We want high-Z to end up at row 0 (top). "Up" is -Y (angle -90 deg).
    float rotationDeg = static_cast<float>(-90.0 - angleDeg);

    // Normalize to [-180, 180] range
    while (rotationDeg > 180.0f) rotationDeg -= 360.0f;
    while (rotationDeg < -180.0f) rotationDeg += 360.0f;

    return rotationDeg;
}

void QuadSurface::orientZUp()
{
    float angle = computeZOrientationAngle();
    if (std::abs(angle) > 0.5f) {
        std::cout << "QuadSurface::orientZUp: Rotating by " << angle
                  << " degrees to place high-Z at top" << std::endl;
        rotate(angle);
    }
}

namespace {
// Helper to flip all layers in a multi-layer TIFF file
void flipMultiLayerTiff(const std::filesystem::path& tiffPath, int flipCode) {
    if (!std::filesystem::exists(tiffPath)) {
        return;
    }

    // Read all layers
    std::vector<cv::Mat> layers;
    if (!cv::imreadmulti(tiffPath.string(), layers, cv::IMREAD_UNCHANGED)) {
        std::cerr << "Warning: Could not read multi-layer TIFF: " << tiffPath << std::endl;
        return;
    }

    if (layers.empty()) {
        return;
    }

    // Flip each layer
    for (auto& layer : layers) {
        cv::flip(layer, layer, flipCode);
    }

    // Write back all layers
    if (!cv::imwritemulti(tiffPath.string(), layers)) {
        std::cerr << "Warning: Could not write flipped multi-layer TIFF: " << tiffPath << std::endl;
    }
}

// Helper to flip a single-layer TIFF file
void flipSingleTiff(const std::filesystem::path& tiffPath, int flipCode) {
    if (!std::filesystem::exists(tiffPath)) {
        return;
    }

    cv::Mat img = cv::imread(tiffPath.string(), cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "Warning: Could not read TIFF: " << tiffPath << std::endl;
        return;
    }

    cv::flip(img, img, flipCode);

    if (!cv::imwrite(tiffPath.string(), img)) {
        std::cerr << "Warning: Could not write flipped TIFF: " << tiffPath << std::endl;
    }
}
} // anonymous namespace

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

    // Flip external TIFF files on disk
    if (!path.empty()) {
        flipMultiLayerTiff(path / "multilayer_mask.tif", flipCode);
        flipSingleTiff(path / "generations.tif", flipCode);
        flipSingleTiff(path / "approval.tif", flipCode);
    }

    // Invalidate cached bbox
    _bbox = {{-1, -1, -1}, {-1, -1, -1}};
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

    // Flip external TIFF files on disk
    if (!path.empty()) {
        flipMultiLayerTiff(path / "multilayer_mask.tif", flipCode);
        flipSingleTiff(path / "generations.tif", flipCode);
        flipSingleTiff(path / "approval.tif", flipCode);
    }

    // Invalidate cached bbox
    _bbox = {{-1, -1, -1}, {-1, -1, -1}};
}

// Overlapping JSON file utilities
void write_overlapping_json(const std::filesystem::path& seg_path, const std::set<std::string>& overlapping_names) {
    auto overlap_json = utils::Json::object();
    auto arr = utils::Json::array();
    for (const auto& n : overlapping_names)
        arr.push_back(n);
    overlap_json["overlapping"] = std::move(arr);

    std::ofstream o(seg_path / "overlapping.json");
    o << overlap_json.dump(4) << std::endl;
}

std::set<std::string> read_overlapping_json(const std::filesystem::path& seg_path) {
    std::set<std::string> overlapping;
    std::filesystem::path json_path = seg_path / "overlapping.json";

    if (std::filesystem::exists(json_path)) {
        auto overlap_json = utils::Json::parse_file(json_path);

        if (overlap_json.contains("overlapping")) {
            for (const auto& name : overlap_json["overlapping"]) {
                overlapping.insert(name.get_string());
            }
        }
    }

    return overlapping;
}

void QuadSurface::readOverlappingJson()
{
    if (path.empty()) {
        return;
    }
    if (std::filesystem::exists(path / "overlapping")) {
        throw std::runtime_error(
            "Found overlapping directory at: " + (path / "overlapping").string() +
            "\nPlease run overlapping_to_json.py on " + path.parent_path().string() + " to convert it to JSON format"
        );
    }
    _overlappingIds = read_overlapping_json(path);
}

void QuadSurface::writeOverlappingJson() const
{
    if (path.empty()) {
        return;
    }
    write_overlapping_json(path, _overlappingIds);
}

std::optional<std::filesystem::file_time_type> QuadSurface::readMaskTimestamp(const std::filesystem::path& dir)
{
    const std::filesystem::path maskPath = dir / "mask.tif";
    std::error_code ec;
    if (!std::filesystem::exists(maskPath, ec) || ec) {
        return std::nullopt;
    }
    auto ts = std::filesystem::last_write_time(maskPath, ec);
    if (ec) {
        return std::nullopt;
    }
    return ts;
}

void QuadSurface::refreshMaskTimestamp()
{
    _maskTimestamp = readMaskTimestamp(path);
}

// Surface overlap/containment tests
bool overlap(QuadSurface& a, QuadSurface& b, int max_iters)
{
    if (!intersect(a.bbox(), b.bbox()))
        return false;

    cv::Mat_<cv::Vec3f> points = a.rawPoints();
    for(int r=0; r<std::max(10, max_iters/10); r++) {
        cv::Vec2f p = {static_cast<float>(rand() % points.cols), static_cast<float>(rand() % points.rows)};
        cv::Vec3f loc = points(p[1], p[0]);
        if (loc[0] == -1)
            continue;

        cv::Vec3f ptr(0, 0, 0);
        if (b.pointTo(ptr, loc, 2.0, max_iters) <= 2.0) {
            return true;
        }
    }
    return false;
}

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

bool contains_any(QuadSurface& a, const std::vector<cv::Vec3f>& locs)
{
    for(auto& p : locs)
        if (contains(a, p))
            return true;

    return false;
}
