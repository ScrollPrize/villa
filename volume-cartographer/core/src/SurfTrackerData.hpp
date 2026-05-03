#pragma once

#include "vc/core/util/HashFunctions.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <ceres/ceres.h>
#include <opencv2/core.hpp>

#include <cstddef>
#include <set>
#include <unordered_map>
#include <utility>

using SurfPoint = std::pair<QuadSurface *, cv::Vec2i>;

class resId_t
{
public:
    resId_t();
    resId_t(int type, QuadSurface *sm, const cv::Vec2i& p);
    resId_t(int type, QuadSurface *sm, const cv::Vec2i& a, const cv::Vec2i& b);

    bool operator==(const resId_t& o) const;

    int _type;
    QuadSurface *_sm;
    cv::Vec2i _p;
};

struct resId_hash {
    static size_t operator()(resId_t id);
};

struct SurfPoint_hash {
    static size_t operator()(SurfPoint p);
};

// Surface tracking data for loss functions.
class SurfTrackerData
{
public:
    cv::Vec2d& loc(QuadSurface *sm, const cv::Vec2i& loc);
    ceres::ResidualBlockId& resId(const resId_t& id);
    bool hasResId(const resId_t& id) const;
    bool has(QuadSurface *sm, const cv::Vec2i& loc) const;
    void erase(QuadSurface *sm, const cv::Vec2i& loc);
    void eraseSurf(QuadSurface *sm, const cv::Vec2i& loc);
    std::set<QuadSurface *>& surfs(const cv::Vec2i& loc);
    const std::set<QuadSurface *>& surfsC(const cv::Vec2i& loc) const;
    cv::Vec3d lookup_int(QuadSurface *sm, const cv::Vec2i& p);
    bool valid_int(QuadSurface *sm, const cv::Vec2i& p);
    static cv::Vec3d lookup_int_loc(QuadSurface *sm, const cv::Vec2f& l);
    void flip_x(int x0);
    void translate(const cv::Vec2i& delta);

    std::unordered_map<SurfPoint, cv::Vec2d, SurfPoint_hash> _data;
    std::unordered_map<resId_t, ceres::ResidualBlockId, resId_hash> _res_blocks;
    std::unordered_map<cv::Vec2i, std::set<QuadSurface *>, std::vec2i_hash> _surfs;
    std::set<QuadSurface *> _emptysurfs;
    cv::Vec3d seed_coord;
    cv::Vec2i seed_loc;
};
