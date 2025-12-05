#include "vc/core/util/QuadSurface.hpp"

#include "vc/core/util/ChunkCache.hpp"
#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/PointIndex.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include <nlohmann/json.hpp>
#include <system_error>
#include <cmath>
#include <limits>
#include <cerrno>
#include <algorithm>
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

QuadSurface::QuadSurface(const cv::Mat_<cv::Vec3f> &points, const cv::Vec2f &scale) :
    QuadSurface(new cv::Mat_<cv::Vec3f>(points.clone()), scale)
{
}

QuadSurface::QuadSurface(cv::Mat_<cv::Vec3f> *points, const cv::Vec2f &scale)
{
    _points = points;
    //-1 as many times we read with linear interpolation and access +1 locations
    _bounds = {0,0,_points->cols-1,_points->rows-1};
    _scale = scale;
    _center = {static_cast<float>(_points->cols/2.0/_scale[0]), static_cast<float>(_points->rows/2.0/_scale[1]), 0.f};
}

QuadSurface::~QuadSurface()
{
    if (_points) {
        delete _points;
    }
}

cv::Vec3f QuadSurface::pointer()
{
    return cv::Vec3f(0, 0, 0);
}

void QuadSurface::move(cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    ptr += cv::Vec3f(offset[0]*_scale[0], offset[1]*_scale[1], offset[2]);
}

bool QuadSurface::valid(const cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    cv::Vec3f p = internal_loc(offset+_center, ptr, _scale);
    return loc_valid_xy(*_points, {p[0], p[1]});
}


cv::Vec3f QuadSurface::coord(const cv::Vec3f &ptr, const cv::Vec3f &offset)
{
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

cv::Vec3f QuadSurface::normal(const cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    cv::Vec3f p = internal_loc(offset+_center, ptr, _scale);
    return grid_normal((*_points), p);
}

cv::Vec3f QuadSurface::gridNormal(int row, int col) const
{
    if (!_points) {
        return {NAN, NAN, NAN};
    }
    // grid_normal expects (col, row) order in the Vec3f
    return grid_normal(*_points, cv::Vec3f(static_cast<float>(col), static_cast<float>(row), 0.0f));
}

void QuadSurface::setChannel(const std::string& name, const cv::Mat& channel)
{
    _channels[name] = channel;
}

cv::Mat QuadSurface::channel(const std::string& name, int flags)
{
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
    if (!_points) return 0;
    auto range = validPoints();
    return static_cast<int>(std::distance(range.begin(), range.end()));
}

int QuadSurface::countValidQuads() const
{
    if (!_points) return 0;
    auto range = validQuads();
    return static_cast<int>(std::distance(range.begin(), range.end()));
}

cv::Mat_<uint8_t> QuadSurface::validMask() const
{
    if (!_points || _points->empty()) {
        return cv::Mat_<uint8_t>();
    }
    const int rows = _points->rows;
    const int cols = _points->cols;
    cv::Mat_<uint8_t> mask(rows, cols);

#pragma omp parallel for schedule(dynamic, 1)
    for (int j = 0; j < rows; ++j) {
        for (int i = 0; i < cols; ++i) {
            const cv::Vec3f& p = (*_points)(j, i);
            const bool ok = std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]) &&
                           !(p[0] == -1.f && p[1] == -1.f && p[2] == -1.f);
            mask(j, i) = ok ? 255 : 0;
        }
    }
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
        writeTiff(maskPath, mask);
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
}

void QuadSurface::gen(cv::Mat_<cv::Vec3f>* coords,
                      cv::Mat_<cv::Vec3f>* normals,
                      cv::Size size,
                      const cv::Vec3f& ptr,
                      float scale,
                      const cv::Vec3f& offset)
{
    const bool need_normals = (normals != nullptr) || offset[2] || ptr[2];

    const cv::Vec3f ul = internal_loc(offset/scale + _center, ptr, _scale);
    const int w = size.width, h = size.height;

    cv::Mat_<cv::Vec3f> coords_local, normals_local;
    if (!coords)  coords  = &coords_local;
    if (!normals) normals = &normals_local;

    coords->create(size + cv::Size(8, 8));

    // --- build mapping  ---------------------------------
    const double sx = static_cast<double>(_scale[0]) / static_cast<double>(scale);
    const double sy = static_cast<double>(_scale[1]) / static_cast<double>(scale);
    const double ox = static_cast<double>(ul[0]) - 4.0 * sx;
    const double oy = static_cast<double>(ul[1]) - 4.0 * sy;

    std::array<cv::Point2f,3> srcf = {
        cv::Point2f(static_cast<float>(ox),                       static_cast<float>(oy)),
        cv::Point2f(static_cast<float>(ox + (w + 8) * sx),        static_cast<float>(oy)),
        cv::Point2f(static_cast<float>(ox),                       static_cast<float>(oy + (h + 8) * sy))
    };
    std::array<cv::Point2f,3> dstf = {
        cv::Point2f(0.f, 0.f),
        cv::Point2f(static_cast<float>(w + 8), 0.f),
        cv::Point2f(0.f, static_cast<float>(h + 8))
    };

    cv::Mat A = cv::getAffineTransform(srcf.data(), dstf.data());

    // --- build a source validity mask (255 if point is valid) -------------
    cv::Mat valid_src = validMask();

    // --- warp coords with seam-safe border (replicate) -------------------
    cv::Mat_<cv::Vec3f> coords_big;
    cv::warpAffine(*_points, coords_big, A, size + cv::Size(8, 8),
                cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    // --- warp validity with constant 0 (no replicate leakage) -----------
    cv::Mat valid_big;
    cv::warpAffine(valid_src, valid_big, A, size + cv::Size(8, 8),
                cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

    // --- normals: sample on SOURCE grid -------------------
    cv::Mat_<cv::Vec3f> normals_big;
    if (need_normals) {
        normals_big.create(size + cv::Size(8, 8));
        normals_big.setTo(cv::Vec3f(std::numeric_limits<float>::quiet_NaN(),
                                    std::numeric_limits<float>::quiet_NaN(),
                                    std::numeric_limits<float>::quiet_NaN()));
        for (int j = 0; j < h; ++j) {
            const double y = oy + (j + 4.0) * sy;
            for (int i = 0; i < w; ++i) {
                const double x = ox + (i + 4.0) * sx;
                const int jj = j + 4, ii = i + 4;
                if (valid_big.at<uint8_t>(jj, ii)) {
                    normals_big(jj, ii) = grid_normal(*_points,
                        cv::Vec3f(static_cast<float>(x),
                                static_cast<float>(y),
                                0.0f));
                }
            }
        }
    }

    // --- crop away the 4px halo ----------------------------------------
    cv::Rect inner(4, 4, w, h);
    *coords = coords_big(inner).clone();
    cv::Mat valid = valid_big(inner).clone();
    if (need_normals) {
        *normals = normals_big(inner).clone();
    }

    // --- invalidate out-of-footprint pixels (kill GUI leakage) ----------
    const cv::Vec3f qnan(std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN());
    for (int j = 0; j < h; ++j) {
        const uint8_t* mv = valid.ptr<uint8_t>(j);
        for (int i = 0; i < w; ++i) {
            if (!mv[i]) {
                (*coords)(j, i) = qnan;
                if (need_normals) (*normals)(j, i) = qnan;
            }
        }
    }

    // --- apply offset along normals only where normals are valid --------
    if (need_normals && ul[2] != 0.0f) {
        for (int j = 0; j < h; ++j) {
            for (int i = 0; i < w; ++i) {
                const cv::Vec3f n = (*normals)(j, i);
                if (std::isfinite(n[0]) && std::isfinite(n[1]) && std::isfinite(n[2])) {
                    (*coords)(j, i) += n * ul[2];
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

    //TODO check maybe add more search patterns, compare motion estimatino for video compression, x264/x265, ...
    std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,-1},{-1,0},{-1,1},{1,-1},{1,0},{1,1}};
    // std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,0},{1,0}};
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

//search the surface point that is closest to the target coord
float QuadSurface::pointTo(cv::Vec3f &ptr, const cv::Vec3f &tgt, float th, int max_iters,
                           SurfacePatchIndex* surfaceIndex, PointIndex* pointIndex)
{
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
        // Use R-tree to find candidate triangles near target
        const float searchRadius = std::max(th * 4.0f, 100.0f);
        Rect3D bounds;
        bounds.low = tgt - cv::Vec3f(searchRadius, searchRadius, searchRadius);
        bounds.high = tgt + cv::Vec3f(searchRadius, searchRadius, searchRadius);

        std::vector<std::pair<int, int>> candidateCells;
        surfaceIndex->forEachTriangle(bounds, this, [&](const SurfacePatchIndex::TriangleCandidate& tri) {
            candidateCells.emplace_back(tri.j, tri.i);
        });

        // Search from each candidate cell
        for (const auto& [row, col] : candidateCells) {
            if (col < 1 || col >= _points->cols - 1 || row < 1 || row >= _points->rows - 1) {
                continue;
            }
            if ((*_points)(row, col)[0] == -1) {
                continue;
            }

            loc = {static_cast<float>(col), static_cast<float>(row)};
            dist = search_min_loc(*_points, loc, _out, tgt, step_small, _scale[0]*0.1);

            if (dist < th && dist >= 0) {
                ptr = cv::Vec3f(loc[0], loc[1], 0) - cv::Vec3f(_center[0]*_scale[0], _center[1]*_scale[1], 0);
                return dist;
            } else if (dist >= 0 && dist < min_dist) {
                min_loc = loc;
                min_dist = dist;
            }
        }

        // If we found something decent with R-tree, return it
        if (min_dist < th * 2.0f) {
            ptr = cv::Vec3f(min_loc[0], min_loc[1], 0) - cv::Vec3f(_center[0]*_scale[0], _center[1]*_scale[1], 0);
            return min_dist;
        }
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

    // Delete from disk
    if (!path.empty()) {
        std::filesystem::path maskFile = path / "mask.tif";
        std::error_code ec;
        std::filesystem::remove(maskFile, ec);
    }
}

void QuadSurface::writeDataToDirectory(const std::filesystem::path& dir, const std::string& skipChannel)
{
    // Split the points matrix into x, y, z channels
    std::vector<cv::Mat> xyz;
    cv::split((*_points), xyz);

    // Write x/y/z as 32-bit float tiled TIFF with LZW
    writeTiff(dir / "x.tif", xyz[0]);
    writeTiff(dir / "y.tif", xyz[1]);
    writeTiff(dir / "z.tif", xyz[2]);

    // OpenCV compression params for fallback
    std::vector<int> compression_params = { cv::IMWRITE_TIFF_COMPRESSION, 5 };

    // Save additional channels
    for (auto const& [name, mat] : _channels) {
        if (!mat.empty() && (skipChannel.empty() || name != skipChannel)) {
            bool wrote = false;

            // Try tiled TIFF for single-channel ancillary data (8U/16U/32F)
            if (mat.channels() == 1 &&
                (mat.type() == CV_8UC1 || mat.type() == CV_16UC1 || mat.type() == CV_32FC1))
            {
                try {
                    writeTiff(dir / (name + ".tif"), mat);
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
    nlohmann::json snapshotMeta;
    if (meta) {
        snapshotMeta = *meta;
    }
    snapshotMeta["bbox"] = {{bbox().low[0], bbox().low[1], bbox().low[2]},
                            {bbox().high[0], bbox().high[1], bbox().high[2]}};
    snapshotMeta["type"] = "seg";
    snapshotMeta["uuid"] = id;
    snapshotMeta["format"] = "tifxyz";
    snapshotMeta["scale"] = {_scale[0], _scale[1]};

    std::ofstream o(snapshot_dest / "meta.json");
    o << std::setw(4) << snapshotMeta << std::endl;
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

    // Save to temporary location first for atomic operation
    std::filesystem::path temp_path = target_path.parent_path() / ".tmp" / target_path.filename();
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
    if (!meta)
        meta = new nlohmann::json;

    (*meta)["bbox"] = {{bbox().low[0], bbox().low[1], bbox().low[2]},
                       {bbox().high[0], bbox().high[1], bbox().high[2]}};
    (*meta)["type"] = "seg";
    (*meta)["uuid"] = uuid;
    (*meta)["format"] = "tifxyz";
    (*meta)["scale"] = {_scale[0], _scale[1]};

    std::ofstream o(path / "meta.json.tmp");
    o << std::setw(4) << (*meta) << std::endl;
    o.close();

    // Rename to make creation atomic
    std::filesystem::rename(path / "meta.json.tmp", path / "meta.json");

    // Atomically move the saved data to the final location
    bool replacedExisting = false;
    if (force_overwrite && std::filesystem::exists(final_path)) {
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
    }

    if (!replacedExisting) {
        std::filesystem::rename(temp_path, final_path);
    }

    // Update the path to the final canonical location
    path = final_path;
    id = uuid;
}

void QuadSurface::save_meta()
{
    if (!meta)
        throw std::runtime_error("can't save_meta() without metadata!");
    if (path.empty())
        throw std::runtime_error("no storage path for QuadSurface");

    std::ofstream o(path/"meta.json.tmp");
    o << std::setw(4) << (*meta) << std::endl;

    //rename to make creation atomic
    std::filesystem::rename(path/"meta.json.tmp", path/"meta.json");
}

Rect3D QuadSurface::bbox()
{
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

QuadSurface *load_quad_from_tifxyz(const std::string &path, int flags)
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
    std::ifstream meta_f((std::filesystem::path(path)/"meta.json").string());
    if (!meta_f.is_open() || !meta_f.good()) {
        throw std::runtime_error("Cannot open meta.json at: " + path);
    }
    nlohmann::json metadata = nlohmann::json::parse(meta_f);
    cv::Vec2f scale = {metadata["scale"][0].get<float>(), metadata["scale"][1].get<float>()};

    cv::Mat_<cv::Vec3f>* points = new cv::Mat_<cv::Vec3f>();
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

    QuadSurface *surf = new QuadSurface(points, scale);
    surf->path = path;
    surf->id   = metadata["uuid"];
    surf->meta = new nlohmann::json(metadata);

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

QuadSurface* surface_diff(QuadSurface* a, QuadSurface* b, float tolerance) {
    cv::Mat_<cv::Vec3f>* diff_points = new cv::Mat_<cv::Vec3f>(a->rawPoints().clone());

    int width = diff_points->cols;
    int height = diff_points->rows;

    if (!intersect(a->bbox(), b->bbox())) {
        return new QuadSurface(diff_points, a->scale());
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

    QuadSurface* result = new QuadSurface(diff_points, a->scale());
    return result;
}

QuadSurface* surface_union(QuadSurface* a, QuadSurface* b, float tolerance) {
    cv::Mat_<cv::Vec3f>* union_points = new cv::Mat_<cv::Vec3f>(a->rawPoints().clone());

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

    return new QuadSurface(union_points, a->scale());
}

QuadSurface* surface_intersection(QuadSurface* a, QuadSurface* b, float tolerance) {
    cv::Mat_<cv::Vec3f>* intersect_points = new cv::Mat_<cv::Vec3f>(a->rawPoints().clone());

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

    return new QuadSurface(intersect_points, a->scale());
}


bool QuadSurface::containsPoint(const cv::Vec3f& point, float tolerance) const {
    // Quick bounding box check
    Rect3D bbox = const_cast<QuadSurface*>(this)->bbox();
    if (point[0] < bbox.low[0] - tolerance || point[0] > bbox.high[0] + tolerance ||
        point[1] < bbox.low[1] - tolerance || point[1] > bbox.high[1] + tolerance ||
        point[2] < bbox.low[2] - tolerance || point[2] > bbox.high[2] + tolerance) {
        return false;
    }

    cv::Rect boundary(1, 1, _points->cols-2, _points->rows-2);
    float tolerance_sq = tolerance * tolerance;

    // Try multiple random starting points
    for (int attempt = 0; attempt < 10; attempt++) {
        cv::Vec2f loc;
        if (attempt == 0) {
            // First attempt: start from center
            loc = cv::Vec2f(_points->cols/2.0f, _points->rows/2.0f);
        } else {
            // Random starting points
            loc = cv::Vec2f(
                1 + (rand() % (_points->cols-3)),
                1 + (rand() % (_points->rows-3))
            );
        }

        if (!boundary.contains(cv::Point(loc))) continue;

        cv::Vec3f val = at_int(*_points, loc);
        if (val[0] == -1) continue;

        // Gradient descent search with interpolation
        float step = std::max(_scale[0], _scale[1]) * 2.0f;
        const float min_step = _scale[0] * 0.1f;
        static const cv::Vec2f search_dirs[] = {{0,-1},{0,1},{-1,0},{1,0},{-1,-1},{-1,1},{1,-1},{1,1}};

        while (step >= min_step) {
            bool improved = false;

            for (const auto& dir : search_dirs) {
                cv::Vec2f test_loc = loc + dir * step;
                if (!boundary.contains(cv::Point(test_loc))) continue;

                cv::Vec3f test_val = at_int(*_points, test_loc);
                if (test_val[0] == -1) continue;

                float dist_sq = sdist(test_val, point);

                // Early return if within tolerance
                if (dist_sq <= tolerance_sq) {
                    return true;
                }

                // Move to better position
                if (dist_sq < sdist(val, point)) {
                    loc = test_loc;
                    val = test_val;
                    improved = true;
                    break;
                }
            }

            if (!improved) {
                step *= 0.5f;
            }
        }
    }

    return false;
}
