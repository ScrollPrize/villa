#include "SurfTrackerData.hpp"
#include "vc/core/util/Geometry.hpp"

#include <iostream>
#include <stdexcept>

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

size_t resId_hash::operator()(const resId_t& id) const
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

size_t SurfPoint_hash::operator()(const SurfPoint& p) const
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
    auto it = _surfs.find(loc);
    return it == _surfs.end() ? _emptysurfs : it->second;
}

cv::Vec3d SurfTrackerData::lookup_int(QuadSurface *sm, const cv::Vec2i& p)
{
    auto it = _data.find(std::make_pair(sm, p));
    if (it == _data.end())
        throw std::runtime_error("error, lookup failed!");
    cv::Vec2d l = it->second;
    const auto points = sm->rawPoints();
    if (!loc_valid(points, l))
        return {-1, -1, -1};

    const float xy[]{float(l[1]), float(l[0])};
    cv::Vec3f position;
    sm->pointSurface()->pointsAt(xy, 1, position.val, nullptr);
    return std::isfinite(position[0]) ? cv::Vec3d(position) : cv::Vec3d(-1, -1, -1);
}

bool SurfTrackerData::valid_int(QuadSurface *sm, const cv::Vec2i& p)
{
    auto it = _data.find(std::make_pair(sm, p));
    if (it == _data.end())
        return false;
    cv::Vec2d l = it->second;
    const auto points = sm->rawPoints();
    return loc_valid(points, l);
}

cv::Vec3d SurfTrackerData::lookup_int_loc(QuadSurface *sm, const cv::Vec2f& l)
{
    const cv::Vec2d loc{
        static_cast<double>(l[0]),
        static_cast<double>(l[1])
    };
    const auto points = sm->rawPoints();
    if (!loc_valid(points, loc))
        return {-1, -1, -1};

    const float xy[]{float(l[1]), float(l[0])};
    cv::Vec3f position;
    sm->pointSurface()->pointsAt(xy, 1, position.val, nullptr);
    return std::isfinite(position[0]) ? cv::Vec3d(position) : cv::Vec3d(-1, -1, -1);
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
