#pragma once

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(containers, xarray.hpp)
#include <opencv2/core.hpp>

#include <vc/core/util/HashFunctions.hpp>
#include <vc/core/util/ChunkCache.hpp>

#include <shared_mutex>
#include <z5/dataset.hxx>

//NOTE depending on request this might load a lot (the whole array) into RAM
// void readInterpolated3D(xt::xarray<uint8_t> &out, z5::Dataset *ds, const xt::xarray<float> &coords, ChunkCache *cache = nullptr);
void readInterpolated3D(cv::Mat_<uint8_t> &out, z5::Dataset *ds, const cv::Mat_<cv::Vec3f> &coords, ChunkCache<uint8_t> *cache = nullptr, bool nearest_neighbor=false);
void readInterpolated3D(cv::Mat_<uint16_t> &out, z5::Dataset *ds, const cv::Mat_<cv::Vec3f> &coords, ChunkCache<uint16_t> *cache = nullptr, bool nearest_neighbor=false);
template <typename T>
void readArea3D(xt::xtensor<T,3,xt::layout_type::column_major> &out, const cv::Vec3i offset, z5::Dataset *ds, ChunkCache<T> *cache) { throw std::runtime_error("missing implementation"); }
void readArea3D(xt::xtensor<uint8_t,3,xt::layout_type::column_major> &out, const cv::Vec3i offset, z5::Dataset *ds, ChunkCache<uint8_t> *cache);
void readArea3D(xt::xtensor<uint16_t,3,xt::layout_type::column_major> &out, const cv::Vec3i offset, z5::Dataset *ds, ChunkCache<uint16_t> *cache);
cv::Mat_<cv::Vec3f> smooth_vc_segmentation(const cv::Mat_<cv::Vec3f> &points);
cv::Mat_<cv::Vec3f> vc_segmentation_calc_normals(const cv::Mat_<cv::Vec3f> &points);
void vc_segmentation_scales(cv::Mat_<cv::Vec3f> points, double &sx, double &sy);
cv::Vec3f grid_normal(const cv::Mat_<cv::Vec3f> &points, const cv::Vec3f &loc);

template <typename E>
E at_int(const cv::Mat_<E> &points, cv::Vec2f p)
{
    int x = p[0];
    int y = p[1];
    float fx = p[0]-x;
    float fy = p[1]-y;
    
    E p00 = points(y,x);
    E p01 = points(y,x+1);
    E p10 = points(y+1,x);
    E p11 = points(y+1,x+1);
    
    E p0 = (1-fx)*p00 + fx*p01;
    E p1 = (1-fx)*p10 + fx*p11;
    
    return (1-fy)*p0 + fy*p1;
}

template<typename T, int C>
//l is [y, x]!
bool loc_valid(const cv::Mat_<cv::Vec<T,C>> &m, const cv::Vec2d &l)
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

template<typename T, int C>
//l is [x, y]!
bool loc_valid_xy(const cv::Mat_<cv::Vec<T,C>> &m, const cv::Vec2d &l)
{
    return loc_valid(m, {l[1],l[0]});
}
