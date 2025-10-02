#include "vc/core/util/Surface.hpp"

#include "vc/core/util/Slicing.hpp"
#include "vc/core/types/ChunkedTensor.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
//TODO remove
#include <opencv2/highgui.hpp>

#include <unordered_map>
#include <nlohmann/json.hpp>
#include <system_error>
#include <cmath>
#include <limits>

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

void write_overlapping_json(const std::filesystem::path& seg_path, const std::set<std::string>& overlapping_names) {
    nlohmann::json overlap_json;
    overlap_json["overlapping"] = std::vector<std::string>(overlapping_names.begin(), overlapping_names.end());

    std::ofstream o(seg_path / "overlapping.json");
    o << std::setw(4) << overlap_json << std::endl;
}

std::set<std::string> read_overlapping_json(const std::filesystem::path& seg_path) {
    std::set<std::string> overlapping;
    std::filesystem::path json_path = seg_path / "overlapping.json";

    if (std::filesystem::exists(json_path)) {
        std::ifstream i(json_path);
        nlohmann::json overlap_json;
        i >> overlap_json;

        if (overlap_json.contains("overlapping")) {
            for (const auto& name : overlap_json["overlapping"]) {
                overlapping.insert(name.get<std::string>());
            }
        }
    }

    return overlapping;
}



//NOTE we have 3 coordiante systems. Nominal (voxel volume) coordinates, internal relative (ptr) coords (where _center is at 0/0) and internal absolute (_points) coordinates where the upper left corner is at 0/0.
static cv::Vec3f internal_loc(const cv::Vec3f &nominal, const cv::Vec3f &internal, const cv::Vec2f &scale)
{
    return internal + cv::Vec3f(nominal[0]*scale[0], nominal[1]*scale[1], nominal[2]);
}

static cv::Vec3f nominal_loc(const cv::Vec3f &nominal, const cv::Vec3f &internal, const cv::Vec2f &scale)
{
    return nominal + cv::Vec3f(internal[0]/scale[0], internal[1]/scale[1], internal[2]);
}

Surface::~Surface()
{
    if (meta) {
        delete meta;
    }
}

PlaneSurface::PlaneSurface(cv::Vec3f origin_, cv::Vec3f normal) : _origin(origin_)
{
    cv::normalize(normal, _normal);
    update();
};

void PlaneSurface::setNormal(cv::Vec3f normal)
{
    cv::normalize(normal, _normal);
    update();
}

void PlaneSurface::setOrigin(cv::Vec3f origin)
{
    _origin = origin;
    update();
}

cv::Vec3f PlaneSurface::origin()
{
    return _origin;
}

float PlaneSurface::pointDist(cv::Vec3f wp)
{
    float plane_off = _origin.dot(_normal);
    float scalarp = wp.dot(_normal) - plane_off /*- _z_off*/;

    return abs(scalarp);
}

void PlaneSurface::setInPlaneRotation(float radians)
{
    _inPlaneRotation = radians;
    update();
}

void PlaneSurface::setAxisAlignedRotationKey(int key)
{
    _axisAlignedRotationKey = key;
}

//given origin and normal, return the normalized vector v which describes a point : origin + v which lies in the plane and maximizes v.x at the cost of v.y,v.z
static cv::Vec3f vx_from_orig_norm(const cv::Vec3f &o, const cv::Vec3f &n)
{
    //impossible
    if (n[1] == 0 && n[2] == 0)
        return {0,0,0};

    //also trivial
    if (n[0] == 0)
        return {1,0,0};

    cv::Vec3f v = {1,0,0};

    if (n[1] == 0) {
        v[1] = 0;
        //either n1 or n2 must be != 0, see first edge case
        v[2] = -n[0]/n[2];
        cv::normalize(v, v, 1,0, cv::NORM_L2);
        return v;
    }

    if (n[2] == 0) {
        //either n1 or n2 must be != 0, see first edge case
        v[1] = -n[0]/n[1];
        v[2] = 0;
        cv::normalize(v, v, 1,0, cv::NORM_L2);
        return v;
    }

    v[1] = -n[0]/(n[1]+n[2]);
    v[2] = v[1];
    cv::normalize(v, v, 1,0, cv::NORM_L2);

    return v;
}

static cv::Vec3f vy_from_orig_norm(const cv::Vec3f &o, const cv::Vec3f &n)
{
    cv::Vec3f v = vx_from_orig_norm({o[1],o[0],o[2]}, {n[1],n[0],n[2]});
    return {v[1],v[0],v[2]};
}

static void vxy_from_normal(cv::Vec3f orig, cv::Vec3f normal, cv::Vec3f &vx, cv::Vec3f &vy)
{
    vx = vx_from_orig_norm(orig, normal);
    vy = vy_from_orig_norm(orig, normal);

    //TODO will there be a jump around the midpoint?
    if (abs(vx[0]) >= abs(vy[1]))
        vy = cv::Mat(normal).cross(cv::Mat(vx));
    else
        vx = cv::Mat(normal).cross(cv::Mat(vy));

    //FIXME probably not the right way to normalize the direction?
    if (vx[0] < 0)
        vx *= -1;
    if (vy[1] < 0)
        vy *= -1;
}

static cv::Vec3f rotateAroundAxis(const cv::Vec3f& vector, const cv::Vec3f& axis, float angle)
{
    if (std::abs(angle) <= std::numeric_limits<float>::epsilon()) {
        return vector;
    }

    cv::Vec3d axis_d(axis[0], axis[1], axis[2]);
    double axis_norm = cv::norm(axis_d);
    if (axis_norm == 0.0) {
        return vector;
    }
    axis_d /= axis_norm;

    cv::Mat R;
    cv::Mat rot_vec = (cv::Mat_<double>(3, 1)
        << axis_d[0] * static_cast<double>(angle),
           axis_d[1] * static_cast<double>(angle),
           axis_d[2] * static_cast<double>(angle));
    cv::Rodrigues(rot_vec, R);

    cv::Mat v = (cv::Mat_<double>(3, 1) << vector[0], vector[1], vector[2]);
    cv::Mat res = R * v;
    return cv::Vec3f(static_cast<float>(res.at<double>(0)),
                     static_cast<float>(res.at<double>(1)),
                     static_cast<float>(res.at<double>(2)));
}

void PlaneSurface::update()
{
    cv::Vec3f vx, vy;

    vxy_from_normal(_origin,_normal,vx,vy);

    if (std::abs(_inPlaneRotation) > std::numeric_limits<float>::epsilon()) {
        vx = rotateAroundAxis(vx, _normal, _inPlaneRotation);
        vy = rotateAroundAxis(vy, _normal, _inPlaneRotation);
    }

    cv::normalize(vx, vx, 1, 0, cv::NORM_L2);
    cv::normalize(vy, vy, 1, 0, cv::NORM_L2);

    _vx = vx;
    _vy = vy;

    std::vector <cv::Vec3f> src = {_origin,_origin+_normal,_origin+_vx,_origin+_vy};
    std::vector <cv::Vec3f> tgt = {{0,0,0},{0,0,1},{1,0,0},{0,1,0}};
    cv::Mat transf;
    cv::Mat inliers;

    cv::estimateAffine3D(src, tgt, transf, inliers, 0.1, 0.99);

    _M = transf({0,0,3,3});
    _T = transf({3,0,1,3});
}

cv::Vec3f PlaneSurface::project(cv::Vec3f wp, float render_scale, float coord_scale)
{
    cv::Vec3d res = _M*cv::Vec3d(wp)+_T;
    res *= render_scale*coord_scale;

    return {res(0), res(1), res(2)};
}

float PlaneSurface::scalarp(cv::Vec3f point) const
{
    return point.dot(_normal) - _origin.dot(_normal);
}



void PlaneSurface::gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset)
{
    bool create_normals = normals || offset[2] || ptr[2];
    cv::Vec3f total_offset = internal_loc(offset/scale, ptr, {1,1});

    int w = size.width;
    int h = size.height;

    cv::Mat_<cv::Vec3f> _coords_header;
    cv::Mat_<cv::Vec3f> _normals_header;

    if (!coords)
        coords = &_coords_header;
    if (!normals)
        normals = &_normals_header;

    coords->create(size);

    if (create_normals)
        normals->create(size);

    const cv::Vec3f vx = _vx;
    const cv::Vec3f vy = _vy;

    float m = 1/scale;
    cv::Vec3f use_origin = _origin + _normal*total_offset[2];

#pragma omp parallel for
    for(int j=0;j<h;j++)
        for(int i=0;i<w;i++) {
            (*coords)(j,i) = vx*(i*m+total_offset[0]) + vy*(j*m+total_offset[1]) + use_origin;
        }
}

cv::Vec3f PlaneSurface::pointer()
{
    return cv::Vec3f(0, 0, 0);
}

void PlaneSurface::move(cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    ptr += offset;
}

cv::Vec3f PlaneSurface::loc(const cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    return ptr + offset;
}

cv::Vec3f PlaneSurface::coord(const cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    cv::Mat_<cv::Vec3f> coords;
    gen(&coords, nullptr, {1,1}, ptr, 1.0, offset);
    return coords(0,0);
}

cv::Vec3f PlaneSurface::normal(const cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    return _normal;
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
    _center = {_points->cols/2.0/_scale[0],_points->rows/2.0/_scale[1],0};
}

QuadSurface::~QuadSurface()
{
    if (_points) {
        delete _points;
    }
}

QuadSurface *smooth_vc_segmentation(QuadSurface *src)
{
    cv::Mat_<cv::Vec3f> points = smooth_vc_segmentation(src->rawPoints());

    double sx, sy;
    vc_segmentation_scales(points, sx, sy);

    return new QuadSurface(points, {sx,sy});
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
    return {_points->cols / _scale[0], _points->rows / _scale[1]};
}

cv::Vec2f QuadSurface::scale() const
{
    return _scale;
}

cv::Vec3f QuadSurface::normal(const cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    cv::Vec3f p = internal_loc(offset+_center, ptr, _scale);
    return grid_normal((*_points), p);
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

        if (!channel.empty() && !(flags & SURF_CHANNEL_NORESIZE)) {
            cv::Mat scaled_channel;
            cv::resize(channel, scaled_channel, _points->size(), 0, 0, cv::INTER_NEAREST);
            return scaled_channel;
        }
        return channel;
    }
    return cv::Mat();
}

void QuadSurface::invalidateCache()
{
    if (_points) {
        _bounds = {0, 0, _points->cols - 1, _points->rows - 1};
        _center = {
            _points->cols / 2.0 / _scale[0],
            _points->rows / 2.0 / _scale[1],
            0
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

    // --- build a source validity mask (1 if point is valid) -------------
    cv::Mat valid_src(_points->size(), CV_8U, cv::Scalar(0));
    for (int y = 0; y < _points->rows; ++y) {
        for (int x = 0; x < _points->cols; ++x) {
            const cv::Vec3f& p = (*_points)(y, x);
            // treat -1/-1/-1 or NaNs as invalid
            const bool ok = std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]) &&
                            !(p[0] == -1.f && p[1] == -1.f && p[2] == -1.f);
            valid_src.at<uint8_t>(y, x) = ok ? 255 : 0;
        }
    }

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

static float tdist(const cv::Vec3f &a, const cv::Vec3f &b, float t_dist)
{
    cv::Vec3f d = a-b;
    float l = sqrt(d.dot(d));

    return abs(l-t_dist);
}

static float tdist_sum(const cv::Vec3f &v, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds)
{
    float sum = 0;
    for(int i=0;i<tgts.size();i++) {
        float d = tdist(v, tgts[i], tds[i]);
        sum += d*d;
    }

    return sum;
}

//search location in points where we minimize error to multiple objectives using iterated local search
//tgts,tds -> distance to some POIs
//plane -> stay on plane
float min_loc(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds, PlaneSurface *plane, float init_step, float min_step)
{
    if (!loc_valid(points, {loc[1],loc[0]})) {
        out = {-1,-1,-1};
        return -1;
    }

    bool changed = true;
    cv::Vec3f val = at_int(points, loc);
    out = val;
    float best = tdist_sum(val, tgts, tds);
    if (plane) {
        float d = plane->pointDist(val);
        best += d*d;
    }
    float res;

    // std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,-1},{-1,0},{-1,1},{1,-1},{1,0},{1,1}};
    std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,0},{1,0}};
    float step = init_step;



    while (changed) {
        changed = false;

        for(auto &off : search) {
            cv::Vec2f cand = loc+off*step;

            if (!loc_valid(points, {cand[1],cand[0]})) {
                // out = {-1,-1,-1};
                // loc = {-1,-1};
                // return -1;
                continue;
            }

            val = at_int(points, cand);
            // std::cout << "at" << cand << val << std::endl;
            res = tdist_sum(val, tgts, tds);
            if (plane) {
                float d = plane->pointDist(val);
                res += d*d;
            }
            if (res < best) {
                // std::cout << res << val << step << cand << "\n";
                changed = true;
                best = res;
                loc = cand;
                out = val;
            }
            // else
                // std::cout << "(" << res << val << step << cand << "\n";
        }

        if (changed)
            continue;

        step *= 0.5;
        changed = true;

        if (step < min_step)
            break;
    }

    // std::cout << "best" << best << out << "\n" <<  std::endl;
    return best;
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
        loc = {1 + (rand() % points.cols-3), 1 + (rand() % points.rows-3)};

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

//search the surface point that is closest to th tgt coord
float QuadSurface::pointTo(cv::Vec3f &ptr, const cv::Vec3f &tgt, float th, int max_iters)
{
    cv::Vec2f loc = cv::Vec2f(ptr[0], ptr[1]) + cv::Vec2f(_center[0]*_scale[0], _center[1]*_scale[1]);
    cv::Vec3f _out;

    cv::Vec2f step_small = {std::max(1.0f,_scale[0]), std::max(1.0f,_scale[1])};
    float min_mul = std::min(0.1*_points->cols/_scale[0], 0.1*_points->rows/_scale[1]);
    cv::Vec2f step_large = {min_mul*_scale[0], min_mul*_scale[1]};

    float dist = search_min_loc(*_points, loc, _out, tgt, step_small, _scale[0]*0.1);

    if (dist < th && dist >= 0) {
        ptr = cv::Vec3f(loc[0], loc[1], 0) - cv::Vec3f(_center[0]*_scale[0], _center[1]*_scale[1], 0);
        return dist;
    }

    cv::Vec2f min_loc = loc;
    float min_dist = dist;
    if (min_dist < 0)
        min_dist = 10*(_points->cols/_scale[0]+_points->rows/_scale[1]);

    int r_full = 0;
    int skip_count = 0;
    for(int r=0; r<10*max_iters && r_full<max_iters; r++) {
        loc = {1 + (rand() % (_points->cols-3)), 1 + (rand() % (_points->rows-3))};

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


SurfaceControlPoint::SurfaceControlPoint(Surface *base, const cv::Vec3f &ptr_, const cv::Vec3f &control)
{
    ptr = ptr_;
    orig_wp = base->coord(ptr_);
    normal = base->normal(ptr_);
    control_point = control;
}

DeltaSurface::DeltaSurface(Surface *base) : _base(base)
{

}

void DeltaSurface::setBase(Surface *base)
{
    _base = base;
}

cv::Vec3f DeltaSurface::pointer()
{
    return _base->pointer();
}

void DeltaSurface::move(cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    _base->move(ptr, offset);
}

bool DeltaSurface::valid(const cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    return _base->valid(ptr, offset);
}

cv::Vec3f DeltaSurface::loc(const cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    return _base->loc(ptr, offset);
}

cv::Vec3f DeltaSurface::coord(const cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    return _base->coord(ptr, offset);
}

cv::Vec3f DeltaSurface::normal(const cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    return _base->normal(ptr, offset);
}

float DeltaSurface::pointTo(cv::Vec3f &ptr, const cv::Vec3f &tgt, float th, int max_iters)
{
    return _base->pointTo(ptr, tgt, th, max_iters);
}

void ControlPointSurface::addControlPoint(const cv::Vec3f &base_ptr, cv::Vec3f control_point)
{
    _controls.push_back(SurfaceControlPoint(this, base_ptr, control_point));
}

void ControlPointSurface::gen(cv::Mat_<cv::Vec3f> *coords_, cv::Mat_<cv::Vec3f> *normals_, cv::Size size, const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset)
{
    std::cout << "corr gen " << _controls.size() << std::endl;
    cv::Mat_<cv::Vec3f> _coords_local;

    cv::Mat_<cv::Vec3f> *coords = coords_;

    if (!coords)
        coords = &_coords_local;

    _base->gen(coords, normals_, size, ptr, scale, offset);

    int w = size.width;
    int h = size.height;
    cv::Rect bounds(0,0,w,h);

    cv::Vec3f upper_left_nominal = nominal_loc(offset/scale, ptr, dynamic_cast<QuadSurface*>(_base)->_scale);

    float z_offset = upper_left_nominal[2];
    upper_left_nominal[2] = 0;

    for(auto p : _controls) {
        cv::Vec3f p_loc = nominal_loc(loc(p.ptr), ptr, dynamic_cast<QuadSurface*>(_base)->_scale) - upper_left_nominal;
        std::cout << p_loc << p_loc*scale << loc(p.ptr) << ptr << std::endl;
        p_loc *= scale;
        cv::Rect roi(p_loc[0]-40, p_loc[1]-40, 80, 80);
        cv::Rect area = roi & bounds;

        PlaneSurface plane(p.control_point, p.normal);
        float delta = plane.scalarp(coord(p.ptr));
        cv::Vec3f move = delta*p.normal;

        std::cout << area << roi << bounds << move << p.control_point << p.normal << coord(p.ptr) << std::endl;

        for(int j=area.y; j<area.y+area.height; j++)
            for(int i=area.x; i<area.x+area.width; i++) {
                float w = sdist(p_loc, cv::Vec3f(i,j,0));
                w = exp(-w/(20*20));
                (*coords)(j,i) += w*move;
            }
    }
}

void ControlPointSurface::setBase(Surface *base)
{
    DeltaSurface::setBase(base);

    assert(dynamic_cast<QuadSurface*>(base));

    //FIXME reset control points?
    std::cout << "ERROR implement search for ControlPointSurface::setBase()" << std::endl;
}

RefineCompSurface::RefineCompSurface(z5::Dataset *ds, ChunkCache *cache, QuadSurface *base)
: DeltaSurface(base)
{
    _ds = ds;
    _cache = cache;
}

void RefineCompSurface::gen(cv::Mat_<cv::Vec3f> *coords_, cv::Mat_<cv::Vec3f> *normals_, cv::Size size, const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset)
{
    cv::Mat_<cv::Vec3f> _coords_local;
    cv::Mat_<cv::Vec3f> _normals_local;

    cv::Mat_<cv::Vec3f> *coords = coords_;
    cv::Mat_<cv::Vec3f> *normals = normals_;

    if (!coords)
        coords = &_coords_local;
    if (!normals)
        normals = &_normals_local;

    _base->gen(coords, normals, size, ptr, scale, offset);

    cv::Mat_<cv::Vec3f> res;
    cv::Mat_<float> transparent(size, 1);
    cv::Mat_<float> blur(size, 0);
    cv::Mat_<float> integ_z(size, 0);

    if (stop < start)
        step = -abs(step);

    for(int n=0; n<=(stop-start)/step; n++) {
        cv::Mat_<uint8_t> slice;
        float off = start + step*n;
        readInterpolated3D(slice, _ds, (*coords+*normals*off)*scale, _cache);

        cv::Mat floatslice;
        slice.convertTo(floatslice, CV_32F, 1/255.0);

        cv::GaussianBlur(floatslice, blur, {7,7}, 0);
        cv::Mat opaq_slice = blur;

        opaq_slice = (opaq_slice-low)/(high-low);
        opaq_slice = cv::min(opaq_slice,1);
        opaq_slice = cv::max(opaq_slice,0);

        cv::Mat joint = transparent.mul(opaq_slice);
        integ_z += joint * off * scale;
        transparent = transparent-joint;
    }

    integ_z /= (1-transparent);

    cv::Mat mul;
    cv::cvtColor(integ_z, mul, cv::COLOR_GRAY2BGR);
    *coords += (*normals).mul(mul+1+offset[2]);
}

//TODO check if this actually works?!
static void set_block(cv::Mat_<uint8_t> &block, const cv::Vec3f &last_loc, const cv::Vec3f &loc, const cv::Rect &roi, float step)
{
    int x1 = (loc[0]-roi.x)/step;
    int y1 = (loc[1]-roi.y)/step;
    int x2 = (last_loc[0]-roi.x)/step;
    int y2 = (last_loc[1]-roi.y)/step;

    if (x1 < 0 || y1 < 0 || x1 >= block.cols || y1 >= block.rows)
        return;
    if (x2 < 0 || y2 < 0 || x2 >= block.cols || y2 >= block.rows)
        return;

    if (x1 == x2 && y1 == y2)
        block(y1, x1) = 1;
    else
        cv::line(block, {x1,y1},{x2,y2}, 3);
}

static uint8_t get_block(const cv::Mat_<uint8_t> &block, const cv::Vec3f &loc, const cv::Rect &roi, float step)
{
    int x = (loc[0]-roi.x)/step;
    int y = (loc[1]-roi.y)/step;

    if (x < 0 || y < 0 || x >= block.cols || y >= block.rows)
        return 1;

    return block(y, x);
}


void find_intersect_segments(std::vector<std::vector<cv::Vec3f>> &seg_vol, std::vector<std::vector<cv::Vec2f>> &seg_grid, const cv::Mat_<cv::Vec3f> &points, PlaneSurface *plane, const cv::Rect &plane_roi, float step, int min_tries)
{
    //start with random points and search for a plane intersection

    float block_step = 0.5*step;

    cv::Mat_<uint8_t> block(cv::Size(plane_roi.width/block_step, plane_roi.height/block_step), 0);

    cv::Rect grid_bounds(1,1,points.cols-2,points.rows-2);

    std::vector<std::vector<cv::Vec3f>> seg_vol_raw;
    std::vector<std::vector<cv::Vec2f>> seg_grid_raw;

    for(int r=0;r<std::max(min_tries, std::max(points.cols,points.rows)/100);r++) {
        std::vector<cv::Vec3f> seg;
        std::vector<cv::Vec2f> seg_loc;
        std::vector<cv::Vec3f> seg2;
        std::vector<cv::Vec2f> seg_loc2;
        cv::Vec2f loc;
        cv::Vec2f loc2;
        cv::Vec2f loc3;
        cv::Vec3f point;
        cv::Vec3f point2;
        cv::Vec3f point3;
        cv::Vec3f plane_loc;
        cv::Vec3f last_plane_loc;
        float dist = -1;


        //initial points
        for(int i=0;i<std::max(min_tries, std::max(points.cols,points.rows)/100);i++) {
            loc = {std::rand() % (points.cols-1), std::rand() % (points.rows-1)};
            point = at_int(points, loc);

            plane_loc = plane->project(point);
            if (!plane_roi.contains(cv::Point(plane_loc[0],plane_loc[1])))
                continue;

                dist = min_loc(points, loc, point, {}, {}, plane, std::min(points.cols,points.rows)*0.1, 0.01);

                plane_loc = plane->project(point);
                if (!plane_roi.contains(cv::Point(plane_loc[0],plane_loc[1])))
                    dist = -1;

                if (get_block(block, plane_loc, plane_roi, block_step))
                    dist = -1;

            if (dist >= 0 && dist <= 1 || !loc_valid_xy(points, loc))
                break;
        }


        if (dist < 0 || dist > 1)
            continue;

        seg.push_back(point);
        seg_loc.push_back(loc);

        //point2
        loc2 = loc;
        //search point at distance of 1 to init point
        dist = min_loc(points, loc2, point2, {point}, {1}, plane, 0.01, 0.0001);

        if (dist < 0 || dist > 1 || !loc_valid_xy(points, loc))
            continue;

        seg.push_back(point2);
        seg_loc.push_back(loc2);

        last_plane_loc = plane->project(point);
        plane_loc = plane->project(point2);
        set_block(block, last_plane_loc, plane_loc, plane_roi, block_step);
        last_plane_loc = plane_loc;

        //go one direction
        for(int n=0;n<100;n++) {
            //now search following points
            cv::Vec2f loc3 = loc2+loc2-loc;

            if (!grid_bounds.contains(cv::Point(loc3)))
                break;

                point3 = at_int(points, loc3);

                //search point close to prediction + dist 1 to last point
                dist = min_loc(points, loc3, point3, {point,point2,point3}, {2*step,step,0}, plane, 0.01, 0.0001);

                //then refine
                dist = min_loc(points, loc3, point3, {point2}, {step}, plane, 0.01, 0.0001);

                if (dist < 0 || dist > 1 || !loc_valid_xy(points, loc))
                    break;

            seg.push_back(point3);
            seg_loc.push_back(loc3);
            point = point2;
            point2 = point3;
            loc = loc2;
            loc2 = loc3;

            plane_loc = plane->project(point3);
            if (get_block(block, plane_loc, plane_roi, block_step))
                break;

            set_block(block, last_plane_loc, plane_loc, plane_roi, block_step);
            last_plane_loc = plane_loc;
        }

        //now the other direction
        loc2 = seg_loc[0];
        loc = seg_loc[1];
        point2 = seg[0];
        point = seg[1];

        last_plane_loc = plane->project(point2);

        //FIXME repeat by not copying code ...
        for(int n=0;n<100;n++) {
            //now search following points
            cv::Vec2f loc3 = loc2+loc2-loc;

            if (!grid_bounds.contains(cv::Point(loc3[0])))
                break;

                point3 = at_int(points, loc3);

                //search point close to prediction + dist 1 to last point
                dist = min_loc(points, loc3, point3, {point,point2,point3}, {2*step,step,0}, plane, 0.01, 0.0001);

                //then refine
                dist = min_loc(points, loc3, point3, {point2}, {step}, plane, 0.01, 0.0001);

                if (dist < 0 || dist > 1 || !loc_valid_xy(points, loc))
                    break;

            seg2.push_back(point3);
            seg_loc2.push_back(loc3);
            point = point2;
            point2 = point3;
            loc = loc2;
            loc2 = loc3;

            plane_loc = plane->project(point3);
            if (get_block(block, plane_loc, plane_roi, block_step))
                break;

            set_block(block, last_plane_loc, plane_loc, plane_roi, block_step);
            last_plane_loc = plane_loc;
        }

        std::reverse(seg2.begin(), seg2.end());
        std::reverse(seg_loc2.begin(), seg_loc2.end());

        seg2.insert(seg2.end(), seg.begin(), seg.end());
        seg_loc2.insert(seg_loc2.end(), seg_loc.begin(), seg_loc.end());


        seg_vol_raw.push_back(seg2);
        seg_grid_raw.push_back(seg_loc2);
    }

    //split up into disconnected segments
    for(int s=0;s<seg_vol_raw.size();s++) {
        std::vector<cv::Vec3f> seg_vol_curr;
        std::vector<cv::Vec2f> seg_grid_curr;
        cv::Vec3f last = {-1,-1,-1};
        for(int n=0;n<seg_vol_raw[s].size();n++) {
                if (last[0] != -1 && cv::norm(last-seg_vol_raw[s][n]) >= 2*step) {
                seg_vol.push_back(seg_vol_curr);
                seg_grid.push_back(seg_grid_curr);
                seg_vol_curr.resize(0);
                seg_grid_curr.resize(0);
            }
            last = seg_vol_raw[s][n];
            seg_vol_curr.push_back(seg_vol_raw[s][n]);
            seg_grid_curr.push_back(seg_grid_raw[s][n]);
        }
        if (seg_vol_curr.size() >= 2) {
            seg_vol.push_back(seg_vol_curr);
            seg_grid.push_back(seg_grid_curr);
        }
    }
}

struct DSReader
{
    z5::Dataset *ds;
    float scale;
    ChunkCache *cache;
};


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

void QuadSurface::save(const std::string &path_, const std::string &uuid, bool force_overwrite)
{
    std::filesystem::path target_path = path_;
    std::filesystem::path final_path = path_;

    if (!force_overwrite && std::filesystem::exists(final_path))
        throw std::runtime_error("path already exists!");


    std::filesystem::path temp_path = target_path.parent_path() / ".tmp" / target_path.filename();
    std::filesystem::create_directories(temp_path.parent_path());
    path = temp_path;

    if (!std::filesystem::create_directories(path)) {
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error("error creating dir for QuadSurface::save(): " + path.string());
        }
    }

    std::vector<cv::Mat> xyz;

    cv::split((*_points), xyz);

    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_TIFF_COMPRESSION);
    compression_params.push_back(5);

    cv::imwrite(path/"x.tif", xyz[0], compression_params);
    cv::imwrite(path/"y.tif", xyz[1], compression_params);
    cv::imwrite(path/"z.tif", xyz[2], compression_params);

    for (auto const& [name, mat] : _channels) {
        if (!mat.empty()) {
            cv::imwrite(path / (name + ".tif"), mat, compression_params);
        }
    }

    if (!meta)
        meta = new nlohmann::json;

    (*meta)["bbox"] = {{bbox().low[0],bbox().low[1],bbox().low[2]},{bbox().high[0],bbox().high[1],bbox().high[2]}};
    (*meta)["type"] = "seg";
    (*meta)["uuid"] = uuid;
    (*meta)["format"] = "tifxyz";
    (*meta)["scale"] = {_scale[0], _scale[1]};
    std::ofstream o(path/"meta.json.tmp");
    o << std::setw(4) << (*meta) << std::endl;

    //rename to make creation atomic
    std::filesystem::rename(path/"meta.json.tmp", path/"meta.json");

    if (force_overwrite && std::filesystem::exists(final_path)) {
        std::filesystem::path versions_path = final_path / "versions";
        std::filesystem::create_directories(versions_path);
        int max_version = -1;
        if (std::filesystem::exists(versions_path)) {
            for (const auto& entry : std::filesystem::directory_iterator(versions_path)) {
                if (entry.is_directory()) {
                    try {
                        int version = std::stoi(entry.path().filename().string());
                        if (version > max_version) {
                            max_version = version;
                        }
                    } catch (const std::invalid_argument& e) {
                        // Ignore non-numeric directory names
                    }
                }
            }
        }
        
        std::filesystem::path new_version_path = versions_path / std::to_string(max_version + 1);

        if (renameat2(AT_FDCWD, temp_path.c_str(), AT_FDCWD, final_path.c_str(), RENAME_EXCHANGE) != 0) {
            const int err = errno;
            if (err == ENOSYS) {
                throw std::runtime_error("Atomic exchange failed: renameat2 syscall with RENAME_EXCHANGE is not supported on this system. This operation requires Linux kernel >= 3.15.");
            } else {
                const std::error_code ec(err, std::generic_category());
                throw std::runtime_error("atomic exchange failed for " + temp_path.string() + " and " + final_path.string() + ": " + ec.message());
            }
        }

        std::filesystem::rename(temp_path/"versions", final_path/"versions");
        std::filesystem::rename(temp_path, new_version_path);
    }
    else
        std::filesystem::rename(temp_path, final_path);
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
    std::vector<cv::Mat_<float>> xyz = {cv::imread(path+"/x.tif",cv::IMREAD_UNCHANGED),cv::imread(path+"/y.tif",cv::IMREAD_UNCHANGED),cv::imread(path+"/z.tif",cv::IMREAD_UNCHANGED)};

    auto points = new cv::Mat_<cv::Vec3f>;
    cv::merge(xyz, (*points));

    std::ifstream meta_f(path+"/meta.json");
    nlohmann::json metadata = nlohmann::json::parse(meta_f);

    cv::Vec2f scale = {metadata["scale"][0].get<float>(), metadata["scale"][1].get<float>()};

    for(int j=0;j<points->rows;j++)
        for(int i=0;i<points->cols;i++)
            //TODO fix this in the patch gen, also check bounds here in general!
            if ((*points)(j,i)[2] <= 0) {
                (*points)(j,i) = {-1,-1,-1};
            }

    if (!(flags & SURF_LOAD_IGNORE_MASK) && std::filesystem::exists(path+"/mask.tif")) {
        std::vector<cv::Mat> layers;
        cv::imreadmulti(path+"/mask.tif", layers, cv::IMREAD_GRAYSCALE);
        cv::Mat_<uint8_t> mask = layers[0];
        cv::resize(mask, mask, points->size(), 0, 0, cv::INTER_NEAREST);
        for(int j=0;j<points->rows;j++)
            for(int i=0;i<points->cols;i++)
                if (!mask(j,i))
                    (*points)(j,i) = {-1,-1,-1};
    }

    QuadSurface *surf = new QuadSurface(points, scale);

    surf->path = path;
    surf->id   = metadata["uuid"];
    surf->meta = new nlohmann::json(metadata);

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

static Rect3D rect_from_json(const nlohmann::json &json)
{
    return {{json[0][0],json[0][1],json[0][2]},{json[1][0],json[1][1],json[1][2]}};
}

bool overlap(SurfaceMeta &a, SurfaceMeta &b, int max_iters)
{
    if (!intersect(a.bbox, b.bbox))
        return false;

    cv::Mat_<cv::Vec3f> points = a.surface()->rawPoints();
    for(int r=0; r<std::max(10, max_iters/10); r++) {
        cv::Vec2f p = {rand() % points.cols, rand() % points.rows};
        cv::Vec3f loc = points(p[1], p[0]);
        if (loc[0] == -1)
            continue;

        cv::Vec3f ptr = b.surface()->pointer();
        if (b.surface()->pointTo(ptr, loc, 2.0, max_iters) <= 2.0) {
            return true;
        }
    }
    return false;
}


bool contains(SurfaceMeta &a, const cv::Vec3f &loc, int max_iters)
{
    if (!intersect(a.bbox, {loc,loc}))
        return false;

    cv::Vec3f ptr = a.surface()->pointer();
    if (a.surface()->pointTo(ptr, loc, 2.0, max_iters) <= 2.0) {
        return true;
    }
    return false;
}

bool contains(SurfaceMeta &a, const std::vector<cv::Vec3f> &locs)
{
    for(auto &p : locs)
        if (!contains(a, p))
            return false;

    return true;
}

bool contains_any(SurfaceMeta &a, const std::vector<cv::Vec3f> &locs)
{
    for(auto &p : locs)
        if (contains(a, p))
            return true;

    return false;
}

SurfaceMeta::SurfaceMeta(const std::filesystem::path &path_, const nlohmann::json &json) : path(path_)
{
    if (json.contains("bbox"))
        bbox = rect_from_json(json["bbox"]);
    meta = new nlohmann::json;
    *meta = json;
}

SurfaceMeta::SurfaceMeta(const std::filesystem::path &path_) : path(path_)
{
    std::ifstream meta_f(path_/"meta.json");
    if (!meta_f.is_open() || !meta_f.good()) {
        throw std::runtime_error("Cannot open meta.json file at: " + path_.string());
    }

    meta = new nlohmann::json;
    try {
        *meta = nlohmann::json::parse(meta_f);
    } catch (const nlohmann::json::parse_error& e) {
        delete meta;
        meta = nullptr;
        throw std::runtime_error("Invalid JSON in meta.json at: " + path_.string() + " - " + e.what());
    }

    if (meta->contains("bbox"))
        bbox = rect_from_json((*meta)["bbox"]);
}

SurfaceMeta::~SurfaceMeta()
{
    if (_surf) {
        delete _surf;
    }

    if (meta) {
        delete meta;
    }
}

void SurfaceMeta::readOverlapping()
{
    if (std::filesystem::exists(path / "overlapping")) {
        throw std::runtime_error(
            "Found overlapping directory at: " + (path / "overlapping").string() +
            "\nPlease run overlapping_to_json.py on " +  path.parent_path().string() + " to convert it to JSON format"
        );
    }
    overlapping_str = read_overlapping_json(path);
}

QuadSurface *SurfaceMeta::surface()
{
    if (!_surf)
        _surf = load_quad_from_tifxyz(path);
    return _surf;
}

void SurfaceMeta::setSurface(QuadSurface *surf)
{
    _surf = surf;
}

std::string SurfaceMeta::name()
{
    return path.filename();
}

void generate_mask(QuadSurface* surf,
                            cv::Mat_<uint8_t>& mask,
                            cv::Mat_<uint8_t>& img,
                            z5::Dataset* ds_high,
                            z5::Dataset* ds_low,
                            ChunkCache* cache) {
    cv::Mat_<cv::Vec3f> points = surf->rawPoints();

    // Choose resolution based on surface size
    if (points.cols >= 4000) {
        // Large surface: work at 0.25x scale
        if (ds_low && cache) {
            readInterpolated3D(img, ds_low, points * 0.25, cache);
        } else {
            img.create(points.size());
            img.setTo(0);
        }

        mask.create(img.size());
        for(int j = 0; j < img.rows; j++) {
            for(int i = 0; i < img.cols; i++) {
                mask(j,i) = (points(j,i)[0] == -1) ? 0 : 255;
            }
        }
    } else {
        // Small surface: resize and downsample
        cv::Mat_<cv::Vec3f> scaled;
        cv::Vec2f scale = surf->scale();
        cv::resize(points, scaled, {0,0}, 1.0/scale[0], 1.0/scale[1], cv::INTER_CUBIC);

        if (ds_high && cache) {
            readInterpolated3D(img, ds_high, scaled, cache);
            cv::resize(img, img, {0,0}, 0.25, 0.25, cv::INTER_CUBIC);
        } else {
            img.create(cv::Size(points.cols/4.0, points.rows/4.0));
            img.setTo(0);
        }

        mask.create(img.size());
        for(int j = 0; j < img.rows; j++) {
            for(int i = 0; i < img.cols; i++) {
                int orig_j = j * 4 * scale[1];
                int orig_i = i * 4 * scale[0];
                mask(j,i) = (points(orig_j, orig_i)[0] == -1) ? 0 : 255;
            }
        }
    }
}

QuadSurface* surface_diff(QuadSurface* a, QuadSurface* b, float tolerance) {
    cv::Mat_<cv::Vec3f>* diff_points = new cv::Mat_<cv::Vec3f>(a->rawPoints().clone());

    int width = diff_points->cols;
    int height = diff_points->rows;

    if (!intersect(a->bbox(), b->bbox())) {
        return new QuadSurface(diff_points, a->scale());
    }

    int removed_count = 0;
    int total_valid = 0;

    #pragma omp parallel for reduction(+:removed_count,total_valid)
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            cv::Vec3f point = (*diff_points)(j, i);

            if (point[0] == -1 && point[1] == -1 && point[2] == -1) {
                continue;
            }

            total_valid++;

            cv::Vec3f ptr = {0,0,0};
            float dist = b->pointTo(ptr, point, tolerance, 100);

            if (dist >= 0 && dist <= tolerance) {
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

    cv::Mat_<cv::Vec3f> b_points = b->rawPoints();

    int added_count = 0;

    // Add points from b that don't exist in a
    for (int j = 0; j < b_points.rows; j++) {
        for (int i = 0; i < b_points.cols; i++) {
            cv::Vec3f point_b = b_points(j, i);

            // Skip invalid points
            if (point_b[0] == -1) {
                continue;
            }

            // Check if this point exists in a
            cv::Vec2f loc_a;
            float dist = pointTo(loc_a, *union_points, point_b, tolerance, 10, a->scale()[0]);

            // If point is not found in a, we need to add it
            if (dist < 0 || dist > tolerance) {
                int grid_x = std::round(i * b->scale()[0] / a->scale()[0]);
                int grid_y = std::round(j * b->scale()[1] / a->scale()[1]);

                if (grid_x >= 0 && grid_x < union_points->cols &&
                    grid_y >= 0 && grid_y < union_points->rows) {

                    if ((*union_points)(grid_y, grid_x)[0] == -1) {
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

    cv::Mat_<cv::Vec3f> b_points = b->rawPoints();

    int kept_count = 0;
    int total_valid = 0;

    // Keep only points that exist in both surfaces
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            cv::Vec3f point_a = (*intersect_points)(j, i);

            // Skip invalid points
            if (point_a[0] == -1) {
                continue;
            }

            total_valid++;

            // Check if this point exists in b
            cv::Vec2f loc_b;
            float dist = pointTo(loc_b, b_points, point_a, tolerance, 10, b->scale()[0]);

            if (dist >= 0 && dist <= tolerance) {
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
