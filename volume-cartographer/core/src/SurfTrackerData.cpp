#include "SurfTrackerData.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

cv::Vec3f at_int_inv(const cv::Mat_<cv::Vec3f>& points, cv::Vec2f p)
{
    int x = p[1];
    int y = p[0];
    float fx = p[1] - x;
    float fy = p[0] - y;

    const cv::Vec3f& p00 = points(y, x);
    const cv::Vec3f& p01 = points(y, x + 1);
    const cv::Vec3f& p10 = points(y + 1, x);
    const cv::Vec3f& p11 = points(y + 1, x + 1);

    cv::Vec3f p0 = (1 - fx) * p00 + fx * p01;
    cv::Vec3f p1 = (1 - fx) * p10 + fx * p11;

    return (1 - fy) * p0 + fy * p1;
}

} // namespace

resId_t::resId_t() : _type(0), _sm(nullptr)
{
}

resId_t::resId_t(int type, QuadSurface *sm, const cv::Vec2i& p) : _type(type), _sm(sm), _p(p)
{
}

resId_t::resId_t(int type, QuadSurface *sm, const cv::Vec2i& a, const cv::Vec2i& b) : _type(type), _sm(sm)
{
    if (a[0] == b[0]) {
        if (a[1] <= b[1])
            _p = a;
        else
            _p = b;
    }
    else if (a[0] < b[0])
        _p = a;
    else
        _p = b;
}

bool resId_t::operator==(const resId_t& o) const
{
    if (_type != o._type)
        return false;
    if (_sm != o._sm)
        return false;
    if (_p != o._p)
        return false;
    return true;
}

size_t resId_hash::operator()(resId_t id)
{
    size_t hash1 = std::hash<int>{}(id._type);
    size_t hash2 = std::hash<void*>{}(id._sm);
    size_t hash3 = std::hash<int>{}(id._p[0]);
    size_t hash4 = std::hash<int>{}(id._p[1]);

    // magic numbers from boost. should be good enough
    size_t hash = hash1 ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
    hash = hash ^ (hash3 + 0x9e3779b9 + (hash << 6) + (hash >> 2));
    hash = hash ^ (hash4 + 0x9e3779b9 + (hash << 6) + (hash >> 2));

    return hash;
}

size_t SurfPoint_hash::operator()(SurfPoint p)
{
    size_t hash1 = std::hash<void*>{}(p.first);
    size_t hash2 = std::hash<int>{}(p.second[0]);
    size_t hash3 = std::hash<int>{}(p.second[1]);

    // magic numbers from boost. should be good enough
    size_t hash = hash1 ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
    hash = hash ^ (hash3 + 0x9e3779b9 + (hash << 6) + (hash >> 2));

    return hash;
}

cv::Vec2d& SurfTrackerData::loc(QuadSurface *sm, const cv::Vec2i& loc)
{
    return _data[{sm, loc}];
}

ceres::ResidualBlockId& SurfTrackerData::resId(const resId_t& id)
{
    return _res_blocks[id];
}

bool SurfTrackerData::hasResId(const resId_t& id) const
{
    return _res_blocks.contains(id);
}

bool SurfTrackerData::has(QuadSurface *sm, const cv::Vec2i& loc) const
{
    return _data.contains({sm, loc});
}

void SurfTrackerData::erase(QuadSurface *sm, const cv::Vec2i& loc)
{
    _data.erase({sm, loc});
}

void SurfTrackerData::eraseSurf(QuadSurface *sm, const cv::Vec2i& loc)
{
    _surfs[loc].erase(sm);
}

std::set<QuadSurface *>& SurfTrackerData::surfs(const cv::Vec2i& loc)
{
    return _surfs[loc];
}

const std::set<QuadSurface *>& SurfTrackerData::surfsC(const cv::Vec2i& loc) const
{
    if (!_surfs.contains(loc))
        return _emptysurfs;
    else
        return _surfs.find(loc)->second;
}

cv::Vec3d SurfTrackerData::lookup_int(QuadSurface *sm, const cv::Vec2i& p)
{
    auto id = std::make_pair(sm, p);
    if (!_data.contains(id))
        throw std::runtime_error("error, lookup failed!");
    cv::Vec2d l = loc(sm, p);
    if (l[0] == -1)
        return {-1, -1, -1};
    else {
        cv::Rect bounds = {0, 0, sm->rawPoints().rows - 2, sm->rawPoints().cols - 2};
        cv::Vec2i li = {floor(l[0]), floor(l[1])};
        if (bounds.contains(cv::Point(li)))
            return at_int_inv(sm->rawPoints(), l);
        else
            return {-1, -1, -1};
    }
}

bool SurfTrackerData::valid_int(QuadSurface *sm, const cv::Vec2i& p)
{
    auto id = std::make_pair(sm, p);
    if (!_data.contains(id))
        return false;
    cv::Vec2d l = loc(sm, p);
    if (l[0] == -1)
        return false;
    else {
        cv::Rect bounds = {0, 0, sm->rawPoints().rows - 2, sm->rawPoints().cols - 2};
        cv::Vec2i li = {floor(l[0]), floor(l[1])};
        if (bounds.contains(cv::Point(li)))
        {
            if (sm->rawPoints()(li[0], li[1])[0] == -1)
                return false;
            if (sm->rawPoints()(li[0] + 1, li[1])[0] == -1)
                return false;
            if (sm->rawPoints()(li[0], li[1] + 1)[0] == -1)
                return false;
            if (sm->rawPoints()(li[0] + 1, li[1] + 1)[0] == -1)
                return false;
            return true;
        }
        else
            return false;
    }
}

cv::Vec3d SurfTrackerData::lookup_int_loc(QuadSurface *sm, const cv::Vec2f& l)
{
    if (l[0] == -1)
        return {-1, -1, -1};
    else {
        cv::Rect bounds = {0, 0, sm->rawPoints().rows - 2, sm->rawPoints().cols - 2};
        if (bounds.contains(cv::Point(l)))
            return at_int_inv(sm->rawPoints(), l);
        else
            return {-1, -1, -1};
    }
}

void SurfTrackerData::flip_x(int x0)
{
    std::cout << " src sizes " << _data.size() << " " << _surfs.size() << std::endl;
    SurfTrackerData old = *this;
    _data.clear();
    _res_blocks.clear();
    _surfs.clear();

    for (auto& it : old._data)
        _data[{it.first.first, {it.first.second[0], x0 + x0 - it.first.second[1]}}] = it.second;

    for (auto& it : old._surfs)
        _surfs[{it.first[0], x0 + x0 - it.first[1]}] = it.second;

    std::cout << " flipped sizes " << _data.size() << " " << _surfs.size() << std::endl;
}

void SurfTrackerData::translate(const cv::Vec2i& delta)
{
    if (delta == cv::Vec2i(0, 0)) {
        return;
    }

    SurfTrackerData old = *this;
    _data.clear();
    _res_blocks.clear();
    _surfs.clear();

    for (auto& it : old._data) {
        _data[{it.first.first, it.first.second + delta}] = it.second;
    }

    for (auto& it : old._surfs) {
        _surfs[it.first + delta] = it.second;
    }

    seed_loc += delta;
}
