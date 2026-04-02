#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/Geometry.hpp"

#include <opencv2/calib3d.hpp>
#include "utils/Json.hpp"

#include <cmath>
#include <limits>

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__x86_64__)
#include <immintrin.h>
#endif

namespace {

//NOTE we have 3 coordinate systems. Nominal (voxel volume) coordinates, internal relative (ptr) coords (where _center is at 0/0) and internal absolute (_points) coordinates where the upper left corner is at 0/0.
static cv::Vec3f internal_loc(const cv::Vec3f &nominal, const cv::Vec3f &internal, const cv::Vec2f &scale)
{
    return internal + cv::Vec3f(nominal[0]*scale[0], nominal[1]*scale[1], nominal[2]);
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
    if (std::abs(vx[0]) >= std::abs(vy[1]))
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

} // anonymous namespace

PlaneSurface::PlaneSurface() = default;
PlaneSurface::~PlaneSurface() = default;

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

    return std::abs(scalarp);
}

void PlaneSurface::setInPlaneRotation(float radians)
{
    _inPlaneRotation = radians;
    update();
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

    return {static_cast<float>(res(0)), static_cast<float>(res(1)), static_cast<float>(res(2))};
}

float PlaneSurface::scalarp(cv::Vec3f point) const
{
    return point.dot(_normal) - _origin.dot(_normal);
}



void PlaneSurface::gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset) const
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
    // Precompute: row_base = vy*(j*m + off_y) + vx*off_x + origin
    // Inner loop: row_base + vx * (i * m) => start + i * step
    cv::Vec3f use_origin = _origin + _normal*total_offset[2];
    cv::Vec3f x_start = vx * total_offset[0] + use_origin;
    cv::Vec3f vx_step = vx * m;
    cv::Vec3f vy_step = vy * m;
    cv::Vec3f vy_base = vy * total_offset[1];

#pragma omp parallel for
    for(int j=0;j<h;j++) {
        cv::Vec3f row_base = vy_base + vy_step * static_cast<float>(j) + x_start;
        float *row = reinterpret_cast<float*>(coords->ptr<cv::Vec3f>(j));

#if defined(__aarch64__)
        // NEON: process 4 pixels per iteration
        float32x4_t base_x = vdupq_n_f32(row_base[0]);
        float32x4_t base_y = vdupq_n_f32(row_base[1]);
        float32x4_t base_z = vdupq_n_f32(row_base[2]);
        float32x4_t step_x = vdupq_n_f32(vx_step[0]);
        float32x4_t step_y = vdupq_n_f32(vx_step[1]);
        float32x4_t step_z = vdupq_n_f32(vx_step[2]);
        float32x4_t idx4 = {0.f, 1.f, 2.f, 3.f};
        float32x4_t four = vdupq_n_f32(4.f);

        int i = 0;
        for (; i + 3 < w; i += 4) {
            float32x4_t px = vmlaq_f32(base_x, idx4, step_x);
            float32x4_t py = vmlaq_f32(base_y, idx4, step_y);
            float32x4_t pz = vmlaq_f32(base_z, idx4, step_z);
            float32x4x3_t out = {px, py, pz};
            vst3q_f32(row + i * 3, out);
            idx4 = vaddq_f32(idx4, four);
        }
        // Scalar tail
        for (; i < w; i++) {
            float fi = static_cast<float>(i);
            row[i*3+0] = row_base[0] + fi * vx_step[0];
            row[i*3+1] = row_base[1] + fi * vx_step[1];
            row[i*3+2] = row_base[2] + fi * vx_step[2];
        }

#elif defined(__x86_64__)
        // SSE: process 4 pixels per iteration
        __m128 base_x = _mm_set1_ps(row_base[0]);
        __m128 base_y = _mm_set1_ps(row_base[1]);
        __m128 base_z = _mm_set1_ps(row_base[2]);
        __m128 step_x = _mm_set1_ps(vx_step[0]);
        __m128 step_y = _mm_set1_ps(vx_step[1]);
        __m128 step_z = _mm_set1_ps(vx_step[2]);
        __m128 idx4 = _mm_set_ps(3.f, 2.f, 1.f, 0.f);
        __m128 four = _mm_set1_ps(4.f);

        int i = 0;
        for (; i + 3 < w; i += 4) {
            __m128 px = _mm_add_ps(base_x, _mm_mul_ps(idx4, step_x));
            __m128 py = _mm_add_ps(base_y, _mm_mul_ps(idx4, step_y));
            __m128 pz = _mm_add_ps(base_z, _mm_mul_ps(idx4, step_z));
            // SOA -> AOS: [x0..x3],[y0..y3],[z0..z3] -> [x0,y0,z0,x1,y1,z1,...]
            // Produce 3 output registers of 4 floats (12 floats for 4 Vec3f).
            //
            // _mm_shuffle_ps(A, B, _MM_SHUFFLE(d,c,b,a)):
            //   result = [A[a], A[b], B[c], B[d]]
            //
            // Intermediates:
            __m128 xy_lo = _mm_unpacklo_ps(px, py); // [x0,y0,x1,y1]
            __m128 xy_hi = _mm_unpackhi_ps(px, py); // [x2,y2,x3,y3]
            // out0 = [x0, y0, z0, x1]
            //   [x0,y0] from xy_lo[0,1], [z0,x1] via shuffle of pz and px
            __m128 zx = _mm_unpacklo_ps(pz, px);    // [z0,x0,z1,x1]
            __m128 out0 = _mm_shuffle_ps(xy_lo, zx, _MM_SHUFFLE(3,0,1,0));
            // out1 = [y1, z1, x2, y2]
            //   [y1,z1] from yz interleave, [x2,y2] from xy_hi
            __m128 yz = _mm_unpacklo_ps(py, pz);    // [y0,z0,y1,z1]
            __m128 out1 = _mm_shuffle_ps(yz, xy_hi, _MM_SHUFFLE(1,0,3,2));
            // out2 = [z2, x3, y3, z3]
            //   [z2,x3] from zx_hi, [y3,z3] from yz_hi
            __m128 zx_hi = _mm_unpackhi_ps(pz, px); // [z2,x2,z3,x3]
            __m128 yz_hi = _mm_unpackhi_ps(py, pz); // [y2,z2,y3,z3]
            __m128 out2 = _mm_shuffle_ps(zx_hi, yz_hi, _MM_SHUFFLE(3,2,3,0));
            float *dst = row + i * 3;
            _mm_storeu_ps(dst + 0, out0);
            _mm_storeu_ps(dst + 4, out1);
            _mm_storeu_ps(dst + 8, out2);
            idx4 = _mm_add_ps(idx4, four);
        }
        // Scalar tail
        for (; i < w; i++) {
            float fi = static_cast<float>(i);
            row[i*3+0] = row_base[0] + fi * vx_step[0];
            row[i*3+1] = row_base[1] + fi * vx_step[1];
            row[i*3+2] = row_base[2] + fi * vx_step[2];
        }

#else
        // Scalar fallback
        cv::Vec3f cur = row_base;
        cv::Vec3f *rowv = reinterpret_cast<cv::Vec3f*>(row);
        for(int i=0;i<w;i++) {
            rowv[i] = cur;
            cur += vx_step;
        }
#endif
    }
}

void PlaneSurface::move(cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    ptr += offset;
}

cv::Vec3f PlaneSurface::loc(const cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    return ptr + offset;
}

cv::Vec3f PlaneSurface::coord(const cv::Vec3f &ptr, const cv::Vec3f &offset) const
{
    cv::Mat_<cv::Vec3f> coords;
    gen(&coords, nullptr, {1,1}, ptr, 1.0, offset);
    return coords(0,0);
}

cv::Vec3f PlaneSurface::normal(const cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    return _normal;
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

