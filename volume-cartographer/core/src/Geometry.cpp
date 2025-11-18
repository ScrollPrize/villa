#include "vc/core/util/Geometry.hpp"

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(generators, xbuilder.hpp)
#include XTENSORINCLUDE(views, xview.hpp)

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <random>

//somehow opencvs functions are pretty slow
static cv::Vec3f normed(const cv::Vec3f& v)
{
    return v/sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
}

static cv::Vec2f vmin(const cv::Vec2f &a, const cv::Vec2f &b)
{
    return {std::min(a[0],b[0]),std::min(a[1],b[1])};
}

static cv::Vec2f vmax(const cv::Vec2f &a, const cv::Vec2f &b)
{
    return {std::max(a[0],b[0]),std::max(a[1],b[1])};
}

cv::Vec3f grid_normal(const cv::Mat_<cv::Vec3f> &points, const cv::Vec3f &loc)
{
    cv::Vec2f inb_loc = {loc[0], loc[1]};
    //move inside from the grid border so w can access required locations
    inb_loc = vmax(inb_loc, {1.f,1.f});
    inb_loc = vmin(inb_loc, {static_cast<float>(points.cols-3), static_cast<float>(points.rows-3)});

    if (!loc_valid_xy(points, inb_loc))
        return {NAN,NAN,NAN};

    if (!loc_valid_xy(points, inb_loc+cv::Vec2f(1,0)))
        return {NAN,NAN,NAN};
    if (!loc_valid_xy(points, inb_loc+cv::Vec2f(-1,0)))
        return {NAN,NAN,NAN};
    if (!loc_valid_xy(points, inb_loc+cv::Vec2f(0,1)))
        return {NAN,NAN,NAN};
    if (!loc_valid_xy(points, inb_loc+cv::Vec2f(0,-1)))
        return {NAN,NAN,NAN};

    cv::Vec3f xv = normed(at_int(points,inb_loc+cv::Vec2f(1,0))-at_int(points,inb_loc-cv::Vec2f(1,0)));
    cv::Vec3f yv = normed(at_int(points,inb_loc+cv::Vec2f(0,1))-at_int(points,inb_loc-cv::Vec2f(0,1)));

    cv::Vec3f n = yv.cross(xv);

    if (std::isnan(n[0]))
        return {NAN,NAN,NAN};

    return normed(n);
}

template <typename E>
static E at_int_impl(const cv::Mat_<E> &points, const cv::Vec2f& p)
{
    int x = p[0];
    int y = p[1];
    float fx = p[0]-x;
    float fy = p[1]-y;

    const E& p00 = points(y,x);
    const E& p01 = points(y,x+1);
    const E& p10 = points(y+1,x);
    const E& p11 = points(y+1,x+1);

    E p0 = (1-fx)*p00 + fx*p01;
    E p1 = (1-fx)*p10 + fx*p11;

    return (1-fy)*p0 + fy*p1;
}

template<typename T, int C>
static bool loc_valid_impl(const cv::Mat_<cv::Vec<T,C>> &m, const cv::Vec2d &l)
{
    if (l[0] == -1)
        return false;

    cv::Rect bounds = {0, 0, m.rows-2,m.cols-2};
    cv::Vec2i li = {static_cast<int>(floor(l[0])), static_cast<int>(floor(l[1]))};

    if (!bounds.contains(cv::Point(li)))
        return false;

    if (m(li[0],li[1])[0] == -1)
        return false;
    if (m(li[0]+1,li[1])[0] == -1)
        return false;
    if (m(li[0],li[1]+1)[0] == -1)
        return false;
    if (m(li[0]+1,li[1]+1)[0] == -1)
        return false;
    return true;
}

static bool loc_valid_scalar(const cv::Mat_<float> &m, const cv::Vec2d &l)
{
    if (l[0] == -1)
        return false;

    cv::Rect bounds = {0, 0, m.rows-2,m.cols-2};
    cv::Vec2i li = {static_cast<int>(floor(l[0])), static_cast<int>(floor(l[1]))};

    if (!bounds.contains(cv::Point(li)))
        return false;

    if (m(li[0],li[1]) == -1)
        return false;
    if (m(li[0]+1,li[1]) == -1)
        return false;
    if (m(li[0],li[1]+1) == -1)
        return false;
    if (m(li[0]+1,li[1]+1) == -1)
        return false;
    return true;
}

template<typename T, int C>
static bool loc_valid_xy_impl(const cv::Mat_<cv::Vec<T,C>> &m, const cv::Vec2d &l)
{
    return loc_valid_impl(m, {l[1],l[0]});
}

static bool loc_valid_xy_scalar(const cv::Mat_<float> &m, const cv::Vec2d &l)
{
    return loc_valid_scalar(m, {l[1],l[0]});
}

cv::Vec3f at_int(const cv::Mat_<cv::Vec3f> &points, const cv::Vec2f &p) {
    return at_int_impl(points, p);
}

float at_int(const cv::Mat_<float> &points, const cv::Vec2f& p) {
    return at_int_impl(points, p);
}

cv::Vec3d at_int(const cv::Mat_<cv::Vec3d> &points, const cv::Vec2f& p) {
    return at_int_impl(points, p);
}

bool loc_valid(const cv::Mat_<cv::Vec3f> &m, const cv::Vec2d &l) {
    return loc_valid_impl(m, l);
}

bool loc_valid(const cv::Mat_<cv::Vec3d> &m, const cv::Vec2d &l) {
    return loc_valid_impl(m, l);
}

bool loc_valid(const cv::Mat_<float> &m, const cv::Vec2d &l) {
    return loc_valid_scalar(m, l);
}

bool loc_valid_xy(const cv::Mat_<cv::Vec3f> &m, const cv::Vec2d &l) {
    return loc_valid_xy_impl(m, l);
}

bool loc_valid_xy(const cv::Mat_<cv::Vec3d> &m, const cv::Vec2d &l) {
    return loc_valid_xy_impl(m, l);
}

bool loc_valid_xy(const cv::Mat_<float> &m, const cv::Vec2d &l) {
    return loc_valid_xy_scalar(m, l);
}


float tdist(const cv::Vec3f &a, const cv::Vec3f &b, float t_dist)
{
    cv::Vec3f d = a-b;
    float l = sqrtf(d.dot(d));

    return abs(l-t_dist);
}

float tdist_sum(const cv::Vec3f &v, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds)
{
    float sum = 0;
    for(int i=0;i<tgts.size();i++) {
        float d = tdist(v, tgts[i], tds[i]);
        sum += d*d;
    }

    return sum;
}