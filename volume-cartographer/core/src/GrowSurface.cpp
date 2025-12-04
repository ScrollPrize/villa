#include <omp.h>
#include <random>

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "vc/core/util/DateTime.hpp"
#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/GridStore.hpp"
#include "vc/core/util/HashFunctions.hpp"
#include "vc/core/util/NormalGridVolume.hpp"
#include "vc/core/util/OMPThreadPointCollection.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfaceMeta.hpp"
#include "vc/core/util/SurfaceModeling.hpp"
#include "vc/core/util/SurfaceArea.hpp"
#include "vc/core/types/ChunkedTensor.hpp"

#include "vc/core/util/LifeTime.hpp"

#include <fstream>
#include <iostream>
#include <unordered_set>   // [APPROVED] track approved (surface, location) pairs
#include <shared_mutex>
#include <filesystem>
#include <limits>
#include <algorithm>
#include <map>
#include <cmath>
#include <vector>

int static dbg_counter = 0;
// Default values for thresholds Will be configurable through JSON
static float local_cost_inl_th = 0.2;
static float same_surface_th = 2.0;
static float straight_weight = 0.7f;       // Weight for 2D straight line constraints
static float straight_weight_3D = 4.0f;    // Weight for 3D straight line constraints
static float sliding_w_scale = 1.0f;       // Scale factor for sliding window
static float z_loc_loss_w = 0.1f;          // Weight for Z location loss constraints
static float dist_loss_2d_w = 1.0f;        // Weight for 2D distance constraints
static float dist_loss_3d_w = 2.0f;        // Weight for 3D distance constraints
static float straight_min_count = 1.0f;    // Minimum number of straight constraints
static int inlier_base_threshold = 20;     // Starting threshold for inliers

// ---- Deterministic helpers --------------------------------------------------
static inline uint64_t mix64(uint64_t x) {
    // SplitMix64
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
static inline double det_jitter01(int y, int x, uint64_t salt) {
    uint64_t h = mix64((uint64_t(y) << 32) ^ uint64_t(x) ^ salt);
    // map to [0,1)
    const double inv = 1.0 / double(UINT64_C(1) << 53);
    return double(h >> 11) * inv;
}
static inline double det_jitter_symm(int y, int x, uint64_t salt) {
    // map to ~[-1,1)
    return 2.0 * det_jitter01(y, x, salt) - 1.0;
}
// -----------------------------------------------------------------------------

static cv::Vec3f at_int_inv(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f p)
{
    int x = p[1];
    int y = p[0];
    float fx = p[1]-x;
    float fy = p[0]-y;

    const cv::Vec3f& p00 = points(y,x);
    const cv::Vec3f& p01 = points(y,x+1);
    const cv::Vec3f& p10 = points(y+1,x);
    const cv::Vec3f& p11 = points(y+1,x+1);

    cv::Vec3f p0 = (1-fx)*p00 + fx*p01;
    cv::Vec3f p1 = (1-fx)*p10 + fx*p11;

    return (1-fy)*p0 + fy*p1;
}

using SurfPoint = std::pair<SurfaceMeta*,cv::Vec2i>;

// Try to count a quad (top-left index j,i) if its four corners are STATE_LOC_VALID.
// Returns the quad area (voxel^2) if counted; 0 otherwise. Caller must ensure
// any check+set of 'quad_done(j,i)' is synchronized if used from parallel code.
static inline double maybe_quad_area_and_mark(int j, int i,
                                              const cv::Mat_<uint8_t>& state,
                                              const cv::Mat_<cv::Vec3d>& points,
                                              cv::Mat_<uint8_t>& quad_done)
{
    if (j < 0 || i < 0 || j >= quad_done.rows || i >= quad_done.cols) return 0.0;
    if (quad_done(j,i)) return 0.0;
    if ( (state(j,   i  ) & STATE_LOC_VALID) &&
         (state(j,   i+1) & STATE_LOC_VALID) &&
         (state(j+1, i  ) & STATE_LOC_VALID) &&
         (state(j+1, i+1) & STATE_LOC_VALID) )
    {
        const double a = vc::surface::quadAreaVox2(points(j,   i  ),
                                                   points(j,   i+1),
                                                   points(j+1, i  ),
                                                   points(j+1, i+1));
        quad_done(j,i) = 1;
        return a;
    }
    return 0.0;
}

// Deterministic ordering for cv::Vec2i (row-major: y, then x)
struct Vec2iLess {
    bool operator()(const cv::Vec2i& a, const cv::Vec2i& b) const {
        return (a[0] < b[0]) || (a[0] == b[0] && a[1] < b[1]);
    }
};

class resId_t
{
public:
    resId_t() : _type(0), _sm(nullptr) {
    } ;
    resId_t(int type, SurfaceMeta* sm, const cv::Vec2i& p) : _type(type), _sm(sm), _p(p) {};
    resId_t(int type, SurfaceMeta* sm, const cv::Vec2i &a, const cv::Vec2i &b) : _type(type), _sm(sm)
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
    bool operator==(const resId_t &o) const
    {
        if (_type != o._type)
            return false;
        if (_sm != o._sm)
            return false;
        if (_p != o._p)
            return false;
        return true;
    }

    int _type;
    SurfaceMeta* _sm;
    cv::Vec2i _p;
};

struct resId_hash {
    static size_t operator()(resId_t id)
    {
        size_t hash1 = std::hash<int>{}(id._type);
        size_t hash2 = std::hash<void*>{}(id._sm);
        size_t hash3 = std::hash<int>{}(id._p[0]);
        size_t hash4 = std::hash<int>{}(id._p[1]);

        //magic numbers from boost. should be good enough
        size_t hash = hash1  ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
        hash =  hash  ^ (hash3 + 0x9e3779b9 + (hash << 6) + (hash >> 2));
        hash =  hash  ^ (hash4 + 0x9e3779b9 + (hash << 6) + (hash >> 2));

        return hash;
    }
};


struct SurfPoint_hash {
    static size_t operator()(SurfPoint p)
    {
        size_t hash1 = std::hash<void*>{}(p.first);
        size_t hash2 = std::hash<int>{}(p.second[0]);
        size_t hash3 = std::hash<int>{}(p.second[1]);

        //magic numbers from boost. should be good enough
        size_t hash = hash1  ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
        hash =  hash  ^ (hash3 + 0x9e3779b9 + (hash << 6) + (hash >> 2));

        return hash;
    }
};

//Surface tracking data for loss functions
class SurfTrackerData
{

public:
    cv::Vec2d &loc(SurfaceMeta *sm, const cv::Vec2i &loc)
    {
        return _data[{sm,loc}];
    }
    ceres::ResidualBlockId &resId(const resId_t &id)
    {
        return _res_blocks[id];
    }
    bool hasResId(const resId_t &id) const
    {
        // std::cout << "check hasResId " << id._sm << " " << id._type << " " << id._p << std::endl;
        return _res_blocks.contains(id);
    }
    bool has(SurfaceMeta *sm, const cv::Vec2i &loc) const {
        return _data.contains({sm,loc});
    }
    void erase(SurfaceMeta *sm, const cv::Vec2i &loc)
    {
        _data.erase({sm,loc});
        clearApprovedPair(sm, loc); // [APPROVED] keep approved set consistent
    }
    void eraseSurf(SurfaceMeta *sm, const cv::Vec2i &loc)
    {
        _surfs[loc].erase(sm);
        clearApprovedPair(sm, loc); // [APPROVED]
    }
    std::set<SurfaceMeta*> &surfs(const cv::Vec2i &loc)
    {
        return _surfs[loc];
    }
    const std::set<SurfaceMeta*> &surfsC(const cv::Vec2i &loc) const
    {
        if (!_surfs.contains(loc))
            return _emptysurfs;
        else
            return _surfs.find(loc)->second;
    }
    cv::Vec3d lookup_int(SurfaceMeta *sm, const cv::Vec2i &p)
    {
        auto id = std::make_pair(sm,p);
        if (!_data.contains(id))
            throw std::runtime_error("error, lookup failed!");
        cv::Vec2d l = loc(sm, p);
        if (l[0] == -1)
            return {-1,-1,-1};
        else {
            cv::Rect bounds = {0, 0, sm->surface()->rawPoints().rows-2,sm->surface()->rawPoints().cols-2};
            cv::Vec2i li = {static_cast<int>(floor(l[0])),static_cast<int>(floor(l[1]))};
            if (bounds.contains(cv::Point(li)))
                return at_int_inv(sm->surface()->rawPoints(), l);
            else
                return {-1,-1,-1};
        }
    }
    bool valid_int(SurfaceMeta *sm, const cv::Vec2i &p)
    {
        auto id = std::make_pair(sm,p);
        if (!_data.contains(id))
            return false;
        cv::Vec2d l = loc(sm, p);
        if (l[0] == -1)
            return false;
        else {
            cv::Rect bounds = {0, 0, sm->surface()->rawPoints().rows-2,sm->surface()->rawPoints().cols-2};
            cv::Vec2i li = {static_cast<int>(floor(l[0])),static_cast<int>(floor(l[1]))};
            if (bounds.contains(cv::Point(li)))
            {
                if (sm->surface()->rawPoints()(li[0],li[1])[0] == -1)
                    return false;
                if (sm->surface()->rawPoints()(li[0]+1,li[1])[0] == -1)
                    return false;
                if (sm->surface()->rawPoints()(li[0],li[1]+1)[0] == -1)
                    return false;
                if (sm->surface()->rawPoints()(li[0]+1,li[1]+1)[0] == -1)
                    return false;
                return true;
            }
            else
                return false;
        }
    }
    static cv::Vec3d lookup_int_loc(SurfaceMeta *sm, const cv::Vec2f &l)
    {
        if (l[0] == -1)
            return {-1,-1,-1};
        else {
            cv::Rect bounds = {0, 0, sm->surface()->rawPoints().rows-2,sm->surface()->rawPoints().cols-2};
            if (bounds.contains(cv::Point(l)))
                return at_int_inv(sm->surface()->rawPoints(), l);
            else
                return {-1,-1,-1};
        }
    }
    void flip_x(int x0)
    {
        std::cout << " src sizes " << _data.size() << " " << _surfs.size() << std::endl;
        SurfTrackerData old = *this;
        _data.clear();
        _res_blocks.clear();
        _surfs.clear();
        _approved.clear(); // [APPROVED]

        for(auto &it : old._data)
            _data[{it.first.first,{it.first.second[0],x0+x0-it.first.second[1]}}] = it.second;

        for(auto &it : old._surfs)
            _surfs[{it.first[0],x0+x0-it.first[1]}] = it.second;

        // [APPROVED] mirror approved pairs
        for (auto &ap : old._approved) {
            const auto& sm = ap.first;
            const auto& p  = ap.second;
            _approved.insert({sm, {p[0], x0 + x0 - p[1]}});
        }

        std::cout << " flipped sizes " << _data.size() << " " << _surfs.size() << std::endl;
    }

    // [APPROVED] helpers
    bool isApproved(SurfaceMeta* sm, const cv::Vec2i& p) const {
        return _approved.contains({sm, p});
    }
    void setApproved(SurfaceMeta* sm, const cv::Vec2i& p) {
        _approved.insert({sm, p});
    }
    void clearApprovedPair(SurfaceMeta* sm, const cv::Vec2i& p) {
        _approved.erase({sm, p});
    }
    void clearApprovedAt(const cv::Vec2i& p) {
        for (auto it = _approved.begin(); it != _approved.end(); ) {
            if (it->second == p) it = _approved.erase(it); else ++it;
        }
    }
    bool approvedAt(const cv::Vec2i& p) const {
        const auto& S = surfsC(p);
        for (auto s : S) if (isApproved(s, p)) return true;
        return false;
    }

// protected:
    std::unordered_map<SurfPoint,cv::Vec2d,SurfPoint_hash> _data;
    std::unordered_map<resId_t,ceres::ResidualBlockId,resId_hash> _res_blocks;
    std::unordered_map<cv::Vec2i, std::set<SurfaceMeta*>> _surfs;
    std::set<SurfaceMeta*> _emptysurfs;
    // [APPROVED] store approved (surface, location) pairs
    std::unordered_set<SurfPoint, SurfPoint_hash> _approved;
    cv::Vec3d seed_coord;
    cv::Vec2i seed_loc;
};

static void copy(const SurfTrackerData &src, SurfTrackerData &tgt, const cv::Rect &roi_)
{
    cv::Rect roi(roi_.y,roi_.x,roi_.height,roi_.width);

    {
        auto it = tgt._data.begin();
        while (it != tgt._data.end()) {
            if (roi.contains(cv::Point(it->first.second)))
                it = tgt._data.erase(it);
            else
                ++it;
        }
    }

    {
        auto it = tgt._surfs.begin();
        while (it != tgt._surfs.end()) {
            if (roi.contains(cv::Point(it->first)))
                it = tgt._surfs.erase(it);
            else
                ++it;
        }
    }

    // [APPROVED] erase approved pairs inside ROI in target
    {
        auto it = tgt._approved.begin();
        while (it != tgt._approved.end()) {
            if (roi.contains(cv::Point(it->second)))
                it = tgt._approved.erase(it);
            else
                ++it;
        }
    }

    for(auto &it : src._data)
        if (roi.contains(cv::Point(it.first.second)))
            tgt._data[it.first] = it.second;
    for(auto &it : src._surfs)
        if (roi.contains(cv::Point(it.first)))
            tgt._surfs[it.first] = it.second;
    // [APPROVED] copy approved pairs inside ROI
    for (auto &ap : src._approved)
        if (roi.contains(cv::Point(ap.second)))
            tgt._approved.insert(ap);

    // tgt.seed_loc = src.seed_loc;
    // tgt.seed_coord = src.seed_coord;
}

static int add_surftrack_distloss(SurfaceMeta *sm, const cv::Vec2i &p, const cv::Vec2i &off, SurfTrackerData &data,
    ceres::Problem &problem, const cv::Mat_<uint8_t> &state, float unit, int flags = 0, ceres::ResidualBlockId *res = nullptr, float w = 1.0)
{
    if ((state(p) & STATE_LOC_VALID) == 0 || !data.has(sm, p))
        return 0;
    if ((state(p+off) & STATE_LOC_VALID) == 0 || !data.has(sm, p+off))
        return 0;

    // Use the global parameter if w is default value (1.0), otherwise use the provided value
    float weight = (w == 1.0f) ? dist_loss_2d_w : w;
    ceres::ResidualBlockId tmp = problem.AddResidualBlock(DistLoss2D::Create(unit*cv::norm(off), weight), nullptr, &data.loc(sm, p)[0], &data.loc(sm, p+off)[0]);
    if (res)
        *res = tmp;

    if ((flags & OPTIMIZE_ALL) == 0)
        problem.SetParameterBlockConstant(&data.loc(sm, p+off)[0]);

    return 1;
}


static int add_surftrack_distloss_3D(cv::Mat_<cv::Vec3d> &points, const cv::Vec2i &p, const cv::Vec2i &off, ceres::Problem &problem,
    const cv::Mat_<uint8_t> &state, float unit, int flags = 0, ceres::ResidualBlockId *res = nullptr, float w = 2.0)
{
    if ((state(p) & (STATE_COORD_VALID | STATE_LOC_VALID)) == 0)
        return 0;
    if ((state(p+off) & (STATE_COORD_VALID | STATE_LOC_VALID)) == 0)
        return 0;

    // Use the global parameter if w is default value (2.0), otherwise use the provided value
    float weight = (w == 2.0f) ? dist_loss_3d_w : w;
    ceres::ResidualBlockId tmp = problem.AddResidualBlock(DistLoss::Create(unit*cv::norm(off), weight), nullptr, &points(p)[0], &points(p+off)[0]);

    // std::cout << cv::norm(points(p)-points(p+off)) << " tgt " << unit << points(p) << points(p+off) << std::endl;
    if (res)
        *res = tmp;

    if ((flags & OPTIMIZE_ALL) == 0)
        problem.SetParameterBlockConstant(&points(p+off)[0]);

    return 1;
}

static int cond_surftrack_distloss_3D(int type, SurfaceMeta *sm, cv::Mat_<cv::Vec3d> &points, const cv::Vec2i &p, const cv::Vec2i &off,
    SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state, float unit, int flags = 0)
{
    resId_t id(type, sm, p, p+off);
    if (data.hasResId(id))
        return 0;

    ceres::ResidualBlockId res;
    int count = add_surftrack_distloss_3D(points, p, off, problem, state, unit, flags, &res);

    data.resId(id) = res;

    return count;
}

static int cond_surftrack_distloss(int type, SurfaceMeta *sm, const cv::Vec2i &p, const cv::Vec2i &off, SurfTrackerData &data,
    ceres::Problem &problem, const cv::Mat_<uint8_t> &state, float unit, int flags = 0)
{
    resId_t id(type, sm, p, p+off);
    if (data.hasResId(id))
        return 0;

    add_surftrack_distloss(sm, p, off, data, problem, state, unit, flags, &data.resId(id));

    return 1;
}

static int add_surftrack_straightloss(SurfaceMeta *sm, const cv::Vec2i &p, const cv::Vec2i &o1, const cv::Vec2i &o2, const cv::Vec2i &o3,
    SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state, int flags = 0, float w = 0.7f)
{
    if ((state(p+o1) & STATE_LOC_VALID) == 0 || !data.has(sm, p+o1))
        return 0;
    if ((state(p+o2) & STATE_LOC_VALID) == 0 || !data.has(sm, p+o2))
        return 0;
    if ((state(p+o3) & STATE_LOC_VALID) == 0 || !data.has(sm, p+o3))
        return 0;

    // Always use the global straight_weight for 2D
    w = straight_weight;

    // std::cout << "add straight " << sm << p << o1 << o2 << o3 << std::endl;
    problem.AddResidualBlock(StraightLoss2D::Create(w), nullptr, &data.loc(sm, p+o1)[0], &data.loc(sm, p+o2)[0], &data.loc(sm, p+o3)[0]);

    if ((flags & OPTIMIZE_ALL) == 0) {
        if (o1 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&data.loc(sm, p+o1)[0]);
        if (o2 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&data.loc(sm, p+o2)[0]);
        if (o3 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&data.loc(sm, p+o3)[0]);
    }

    return 1;
}

static int add_surftrack_straightloss_3D(const cv::Vec2i &p, const cv::Vec2i &o1, const cv::Vec2i &o2, const cv::Vec2i &o3, cv::Mat_<cv::Vec3d> &points,
    ceres::Problem &problem, const cv::Mat_<uint8_t> &state, int flags = 0, ceres::ResidualBlockId *res = nullptr, float w = 4.0f)
{
    if ((state(p+o1) & (STATE_LOC_VALID|STATE_COORD_VALID)) == 0)
        return 0;
    if ((state(p+o2) & (STATE_LOC_VALID|STATE_COORD_VALID)) == 0)
        return 0;
    if ((state(p+o3) & (STATE_LOC_VALID|STATE_COORD_VALID)) == 0)
        return 0;

    // Always use the global straight_weight_3D for 3D
    w = straight_weight_3D;

    // std::cout << "add straight " << sm << p << o1 << o2 << o3 << std::endl;
    ceres::ResidualBlockId tmp =
    problem.AddResidualBlock(StraightLoss::Create(w), nullptr, &points(p+o1)[0], &points(p+o2)[0], &points(p+o3)[0]);
    if (res)
        *res = tmp;

    if ((flags & OPTIMIZE_ALL) == 0) {
        if (o1 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&points(p+o1)[0]);
        if (o2 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&points(p+o2)[0]);
        if (o3 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&points(p+o3)[0]);
    }

    return 1;
}

static int cond_surftrack_straightloss_3D(int type, SurfaceMeta *sm, const cv::Vec2i &p, const cv::Vec2i &o1, const cv::Vec2i &o2, const cv::Vec2i &o3,
    cv::Mat_<cv::Vec3d> &points, SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state, int flags = 0)
{
    resId_t id(type, sm, p);
    if (data.hasResId(id))
        return 0;

    ceres::ResidualBlockId res;
    int count = add_surftrack_straightloss_3D(p, o1, o2, o3, points ,problem, state, flags, &res);

    if (count)
        data.resId(id) = res;

    return count;
}

static int add_surftrack_surfloss(SurfaceMeta *sm, const cv::Vec2i& p, SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state,
    cv::Mat_<cv::Vec3d> &points, float step, ceres::ResidualBlockId *res = nullptr, float w = 0.1)
{
    if ((state(p) & STATE_LOC_VALID) == 0 || !data.valid_int(sm, p))
        return 0;

    ceres::ResidualBlockId tmp = problem.AddResidualBlock(SurfaceLossD::Create(sm->surface()->rawPoints(), w), nullptr,
                                                          &points(p)[0], &data.loc(sm, p)[0]);

    if (res)
        *res = tmp;

    return 1;
}

//gen straigt loss given point and 3 offsets
static int cond_surftrack_surfloss(int type, SurfaceMeta *sm, const cv::Vec2i& p, SurfTrackerData &data, ceres::Problem &problem,
    const cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points, float step)
{
    resId_t id(type, sm, p);
    if (data.hasResId(id))
        return 0;

    ceres::ResidualBlockId res;
    int count = add_surftrack_surfloss(sm, p, data, problem, state, points, step, &res);

    if (count)
        data.resId(id) = res;

    return count;
}

//will optimize only the center point
static int surftrack_add_local(SurfaceMeta *sm, const cv::Vec2i& p, SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points, float step, float src_step, int flags = 0, int *straigh_count_ptr = nullptr)
{
    int count = 0;
    int count_straight = 0;
    //direct
    if (flags & LOSS_3D_INDIRECT) {
        // h
        count += add_surftrack_distloss_3D(points, p, {0,1}, problem, state, step*src_step, flags);
        count += add_surftrack_distloss_3D(points, p, {1,0}, problem, state, step*src_step, flags);
        count += add_surftrack_distloss_3D(points, p, {0,-1}, problem, state, step*src_step, flags);
        count += add_surftrack_distloss_3D(points, p, {-1,0}, problem, state, step*src_step, flags);

        //v
        count += add_surftrack_distloss_3D(points, p, {1,1}, problem, state, step*src_step, flags);
        count += add_surftrack_distloss_3D(points, p, {1,-1}, problem, state, step*src_step, flags);
        count += add_surftrack_distloss_3D(points, p, {-1,1}, problem, state, step*src_step, flags);
        count += add_surftrack_distloss_3D(points, p, {-1,-1}, problem, state, step*src_step, flags);

        //horizontal
        count_straight += add_surftrack_straightloss_3D(p, {0,-2},{0,-1},{0,0}, points, problem, state);
        count_straight += add_surftrack_straightloss_3D(p, {0,-1},{0,0},{0,1}, points, problem, state);
        count_straight += add_surftrack_straightloss_3D(p, {0,0},{0,1},{0,2}, points, problem, state);

        //vertical
        count_straight += add_surftrack_straightloss_3D(p, {-2,0},{-1,0},{0,0}, points, problem, state);
        count_straight += add_surftrack_straightloss_3D(p, {-1,0},{0,0},{1,0}, points, problem, state);
        count_straight += add_surftrack_straightloss_3D(p, {0,0},{1,0},{2,0}, points, problem, state);
    }
    else {
        count += add_surftrack_distloss(sm, p, {0,1}, data, problem, state, step);
        count += add_surftrack_distloss(sm, p, {1,0}, data, problem, state, step);
        count += add_surftrack_distloss(sm, p, {0,-1}, data, problem, state, step);
        count += add_surftrack_distloss(sm, p, {-1,0}, data, problem, state, step);

        //diagonal
        count += add_surftrack_distloss(sm, p, {1,1}, data, problem, state, step);
        count += add_surftrack_distloss(sm, p, {1,-1}, data, problem, state, step);
        count += add_surftrack_distloss(sm, p, {-1,1}, data, problem, state, step);
        count += add_surftrack_distloss(sm, p, {-1,-1}, data, problem, state, step);

        //horizontal
        count_straight += add_surftrack_straightloss(sm, p, {0,-2},{0,-1},{0,0}, data, problem, state);
        count_straight += add_surftrack_straightloss(sm, p, {0,-1},{0,0},{0,1}, data, problem, state);
        count_straight += add_surftrack_straightloss(sm, p, {0,0},{0,1},{0,2}, data, problem, state);

        //vertical
        count_straight += add_surftrack_straightloss(sm, p, {-2,0},{-1,0},{0,0}, data, problem, state);
        count_straight += add_surftrack_straightloss(sm, p, {-1,0},{0,0},{1,0}, data, problem, state);
        count_straight += add_surftrack_straightloss(sm, p, {0,0},{1,0},{2,0}, data, problem, state);
    }

    if (flags & LOSS_ZLOC)
        problem.AddResidualBlock(ZLocationLoss<cv::Vec3f>::Create(
            sm->surface()->rawPoints(),
            data.seed_coord[2] - (p[0]-data.seed_loc[0])*step*src_step, z_loc_loss_w),
            new ceres::HuberLoss(1.0), &data.loc(sm, p)[0]);

    if (flags & SURF_LOSS) {
        count += add_surftrack_surfloss(sm, p, data, problem, state, points, step);
    }

    if (straigh_count_ptr)
        *straigh_count_ptr += count_straight;

    return count + count_straight;
}

//will optimize only the center point
static int surftrack_add_global(SurfaceMeta *sm, const cv::Vec2i& p, SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state,
    cv::Mat_<cv::Vec3d> &points, float step, int flags = 0, float step_onsurf = 0)
{
    if ((state(p) & (STATE_LOC_VALID | STATE_COORD_VALID)) == 0)
        return 0;

    int count = 0;
    //losses are defind in 3D
    if (flags & LOSS_3D_INDIRECT) {
        // h
        count += cond_surftrack_distloss_3D(0, sm, points, p, {0,1}, data, problem, state, step, flags);
        count += cond_surftrack_distloss_3D(0, sm, points, p, {1,0}, data, problem, state, step, flags);
        count += cond_surftrack_distloss_3D(1, sm, points, p, {0,-1}, data, problem, state, step, flags);
        count += cond_surftrack_distloss_3D(1, sm, points, p, {-1,0}, data, problem, state, step, flags);

        //v
        count += cond_surftrack_distloss_3D(2, sm, points, p, {1,1}, data, problem, state, step, flags);
        count += cond_surftrack_distloss_3D(2, sm, points, p, {1,-1}, data, problem, state, step, flags);
        count += cond_surftrack_distloss_3D(3, sm, points, p, {-1,1}, data, problem, state, step, flags);
        count += cond_surftrack_distloss_3D(3, sm, points, p, {-1,-1}, data, problem, state, step, flags);

        //horizontal
        count += cond_surftrack_straightloss_3D(4, sm, p, {0,-2},{0,-1},{0,0}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(4, sm, p, {0,-1},{0,0},{0,1}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(4, sm, p, {0,0},{0,1},{0,2}, points, data, problem, state, flags);

        //vertical
        count += cond_surftrack_straightloss_3D(5, sm, p, {-2,0},{-1,0},{0,0}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(5, sm, p, {-1,0},{0,0},{1,0}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(5, sm, p, {0,0},{1,0},{2,0}, points, data, problem, state, flags);

        //dia1
        count += cond_surftrack_straightloss_3D(6, sm, p, {-2,-2},{-1,-1},{0,0}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(6, sm, p, {-1,-1},{0,0},{1,1}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(6, sm, p, {0,0},{1,1},{2,2}, points, data, problem, state, flags);

        //dia1
        count += cond_surftrack_straightloss_3D(7, sm, p, {-2,2},{-1,1},{0,0}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(7, sm, p, {-1,1},{0,0},{1,-1}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(7, sm, p, {0,0},{1,-1},{2,-2}, points, data, problem, state, flags);
    }

    //losses on surface
    if (flags & LOSS_ON_SURF)
    {
        if (step_onsurf == 0)
            throw std::runtime_error("oops step_onsurf == 0");

        //direct
        count += cond_surftrack_distloss(8, sm, p, {0,1}, data, problem, state, step_onsurf);
        count += cond_surftrack_distloss(8, sm, p, {1,0}, data, problem, state, step_onsurf);
        count += cond_surftrack_distloss(9, sm, p, {0,-1}, data, problem, state, step_onsurf);
        count += cond_surftrack_distloss(9, sm, p, {-1,0}, data, problem, state, step_onsurf);

        //diagonal
        count += cond_surftrack_distloss(10, sm, p, {1,1}, data, problem, state, step_onsurf);
        count += cond_surftrack_distloss(10, sm, p, {1,-1}, data, problem, state, step_onsurf);
        count += cond_surftrack_distloss(11, sm, p, {-1,1}, data, problem, state, step_onsurf);
        count += cond_surftrack_distloss(11, sm, p, {-1,-1}, data, problem, state, step_onsurf);
    }

    if (flags & SURF_LOSS && state(p) & STATE_LOC_VALID)
        count += cond_surftrack_surfloss(14, sm, p, data, problem, state, points, step);

    return count;
}

static double local_cost_destructive(SurfaceMeta *sm, const cv::Vec2i& p, SurfTrackerData &data, cv::Mat_<uint8_t> &state,
    cv::Mat_<cv::Vec3d> &points, float step, float src_step, cv::Vec3f loc, int *ref_count = nullptr, int *straight_count_ptr = nullptr)
{
    uint8_t state_old = state(p);
    state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
    int count;
    int straigh_count;
    if (!straight_count_ptr)
        straight_count_ptr = &straigh_count;

    double test_loss = 0.0;
    {
        ceres::Problem problem_test;

        data.loc(sm, p) = {loc[1], loc[0]};

        count = surftrack_add_local(sm, p, data, problem_test, state, points, step, src_step, 0, straight_count_ptr);
        if (ref_count)
            *ref_count = count;

        problem_test.Evaluate(ceres::Problem::EvaluateOptions(), &test_loss, nullptr, nullptr, nullptr);
    } //destroy problme before data
    data.erase(sm, p);
    state(p) = state_old;

    if (!count)
        return 0;
    else
        return sqrt(test_loss/count);
}


static double local_cost(SurfaceMeta *sm, const cv::Vec2i& p, SurfTrackerData &data, const cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points,
    float step, float src_step, int *ref_count = nullptr, int *straight_count_ptr = nullptr)
{
    int straigh_count;
    if (!straight_count_ptr)
        straight_count_ptr = &straigh_count;

    double test_loss = 0.0;
    ceres::Problem problem_test;

    int count = surftrack_add_local(sm, p, data, problem_test, state, points, step, src_step, 0, straight_count_ptr);
    if (ref_count)
        *ref_count = count;

    problem_test.Evaluate(ceres::Problem::EvaluateOptions(), &test_loss, nullptr, nullptr, nullptr);

    if (!count)
        return 0;
    else
        return sqrt(test_loss/count);
}

static double local_solve(SurfaceMeta *sm, const cv::Vec2i &p, SurfTrackerData &data, const cv::Mat_<uint8_t> &state,
                          cv::Mat_<cv::Vec3d> &points,
                          float step, float src_step, int flags) {
    ceres::Problem problem;

    surftrack_add_local(sm, p, data, problem, state, points, step, src_step, flags);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 10000;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (summary.num_residual_blocks < 3)
        return 10000;

    return summary.final_cost/summary.num_residual_blocks;
}


static inline int surftrack_round_step(float step)
{
    return std::max(1, static_cast<int>(std::lround(step)));
}

static const std::vector<int>& surftrack_tiff_compression_params()
{
    static constexpr int kTiffCompressionLzw = 5;
    static const std::vector<int> params = {
        cv::IMWRITE_TIFF_COMPRESSION,
        kTiffCompressionLzw
    };
    return params;
}

static cv::Mat_<uint16_t> surftrack_generation_channel(
    const cv::Mat_<uint16_t>& generations,
    const cv::Rect& used_area,
    float step)
{
    const int step_int = surftrack_round_step(step);
    if (step_int <= 0 || used_area.width <= 0 || used_area.height <= 0)
        return {};

    cv::Rect bounds(0, 0, generations.cols, generations.rows);
    cv::Rect safe = used_area & bounds;
    if (safe.width <= 0 || safe.height <= 0)
        return {};

    cv::Mat_<uint16_t> channel(safe.height * step_int, safe.width * step_int, static_cast<uint16_t>(0));

    for (int y = 0; y < safe.height; ++y) {
        for (int x = 0; x < safe.width; ++x) {
            uint16_t g = generations(safe.y + y, safe.x + x);
            int base_y = y * step_int;
            int base_x = x * step_int;
            for (int sy = 0; sy < step_int; ++sy)
                for (int sx = 0; sx < step_int; ++sx)
                    channel(base_y + sy, base_x + sx) = g;
        }
    }

    return channel;
}

static uint16_t surftrack_max_generation(const cv::Mat_<uint16_t>& generations,
                                         const cv::Rect& used_area)
{
    cv::Rect bounds(0, 0, generations.cols, generations.rows);
    cv::Rect safe = used_area & bounds;
    if (safe.width <= 0 || safe.height <= 0)
        return 0;

    double min_val = 0.0;
    double max_val = 0.0;
    cv::minMaxLoc(generations(safe), &min_val, &max_val);
    return static_cast<uint16_t>(max_val);
}

static cv::Mat_<cv::Vec3f> surftrack_genpoints_hr(
    SurfTrackerData &data, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points,
    const cv::Rect &used_area, float step, float step_src,
    bool inpaint = false, float approved_weight = 4.0f, bool prefer_approved = true,
    bool parallel = true)
{
    std::cout << "hr_gen: start used_area=" << used_area << " step=" << step
              << " inpaint=" << inpaint << " parallel=" << (parallel?1:0) << std::endl;
    cv::Mat_<cv::Vec3f> points_hr(state.rows*step, state.cols*step, {0,0,0});
    cv::Mat_<float> weights_hr(state.rows*step, state.cols*step, 0.0f);
#pragma omp parallel for if(parallel) //FIXME data access is just not threading friendly ...
    for(int j=used_area.y;j<used_area.br().y-1;j++)
        for(int i=used_area.x;i<used_area.br().x-1;i++) {
            if (state(j,i) & (STATE_LOC_VALID|STATE_COORD_VALID)
                && state(j,i+1) & (STATE_LOC_VALID|STATE_COORD_VALID)
                && state(j+1,i) & (STATE_LOC_VALID|STATE_COORD_VALID)
                && state(j+1,i+1) & (STATE_LOC_VALID|STATE_COORD_VALID))
            {
            for(auto &sm : data.surfsC({j,i})) {
                if (data.valid_int(sm,{j,i})
                    && data.valid_int(sm,{j,i+1})
                    && data.valid_int(sm,{j+1,i})
                    && data.valid_int(sm,{j+1,i+1}))
                {
                    cv::Vec2f l00 = data.loc(sm,{j,i});
                    cv::Vec2f l01 = data.loc(sm,{j,i+1});
                    cv::Vec2f l10 = data.loc(sm,{j+1,i});
                    cv::Vec2f l11 = data.loc(sm,{j+1,i+1});

                    // favor if any corner uses an approved mapping of this surface
                    bool cell_has_approved = prefer_approved && (
                        data.isApproved(sm,{j,i}) ||
                        data.isApproved(sm,{j,i+1}) ||
                        data.isApproved(sm,{j+1,i}) ||
                        data.isApproved(sm,{j+1,i+1})
                    );
                    const float w_surf = (cell_has_approved ? approved_weight : 1.0f);

                    for(int sy=0;sy<=step;sy++)
                        for(int sx=0;sx<=step;sx++) {
                            float fx = sx/step;
                            float fy = sy/step;
                            cv::Vec2f l0 = (1-fx)*l00 + fx*l01;
                            cv::Vec2f l1 = (1-fx)*l10 + fx*l11;
                            cv::Vec2f l = (1-fy)*l0 + fy*l1;
                            if (loc_valid(sm->surface()->rawPoints(), l)) {
                                points_hr(j*step+sy,i*step+sx) += w_surf * cv::Vec3f(SurfTrackerData::lookup_int_loc(sm,l));
                                weights_hr(j*step+sy,i*step+sx) += w_surf;
                            }
                        }
                }
            }
            // [CONNECTIVITY FIX] Inpaint each missing HR sample individually (not just when the center is empty).
            if (inpaint) {
                const cv::Vec3d& c00 = points(j,i);
                const cv::Vec3d& c01 = points(j,i+1);
                const cv::Vec3d& c10 = points(j+1,i);
                const cv::Vec3d& c11 = points(j+1,i+1);

                for(int sy=0;sy<=step;sy++)
                    for(int sx=0;sx<=step;sx++) {
                        if (!weights_hr(j*step+sy,i*step+sx)) {
                            float fx = sx/step;
                            float fy = sy/step;
                            cv::Vec3d c0 = (1-fx)*c00 + fx*c01;
                            cv::Vec3d c1 = (1-fx)*c10 + fx*c11;
                            cv::Vec3d c = (1-fy)*c0 + fy*c1;
                            points_hr(j*step+sy,i*step+sx) = c;
                            weights_hr(j*step+sy,i*step+sx) = 1.0f;
                        }
                    }
            }
        }
    }
#pragma omp parallel for if(parallel)
    for(int j=0;j<points_hr.rows;j++)
        for(int i=0;i<points_hr.cols;i++)
            if (weights_hr(j,i) > 0.0f)
                points_hr(j,i) /= weights_hr(j,i);
            else
                points_hr(j,i) = {-1,-1,-1};

    // snap exact HR node for approved LR nodes AFTER normalization
    // This ensures snapped coords are not subsequently divided by accumulated weights,
    // avoiding distortion/holes at pinned vertices.
#pragma omp parallel for if(parallel)
    for (int j = used_area.y; j < used_area.br().y; ++j)
        for (int i = used_area.x; i < used_area.br().x; ++i)
            if ((state(j,i) & (STATE_LOC_VALID | STATE_COORD_VALID)) && data.approvedAt({j,i}))
                points_hr(j*step, i*step) = cv::Vec3f(points(j,i));

    std::cout << "hr_gen: done" << std::endl;
    return points_hr;
}




//try flattening the current surface mapping assuming direct 3d distances
//this is basically just a reparametrization
static void optimize_surface_mapping(SurfTrackerData &data, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points, cv::Mat_<uint16_t> &generations, cv::Rect used_area,
    cv::Rect static_bounds, float step, float src_step, const cv::Vec2i &seed, int closing_r, bool keep_inpainted = false,
    const std::filesystem::path& tgt_dir = std::filesystem::path(),
    bool pin_approved = true, float approved_weight_hr = 4.0f, bool prefer_approved_in_hr = true,
    bool keep_approved_on_consistency = true,
    int hr_attach_lr_radius = 1,
    float hr_attach_relax_factor = 2.0f,
    bool hr_gen_parallel = true,
    bool remap_parallel = false)
{
    std::cout << "optimizer: optimizing surface " << state.size() << " " << used_area <<  " " << static_bounds << std::endl;

    const int step_int = surftrack_round_step(step);

    cv::Mat_<cv::Vec3d> points_new = points.clone();
    SurfaceMeta sm;
    sm._surf = new QuadSurface(points, {1,1});

    std::shared_mutex mutex;

    SurfTrackerData data_new;
    data_new._data = data._data;

    used_area = cv::Rect(used_area.x-2,used_area.y-2,used_area.size().width+4,used_area.size().height+4);
    // Clamp expanded used_area to valid grid bounds to avoid OOB in subsequent loops
    {
        cv::Rect grid_bounds(0, 0, state.cols, state.rows);
        used_area = used_area & grid_bounds;
    }
    cv::Rect used_area_hr = {used_area.x*step_int, used_area.y*step_int, used_area.width*step_int, used_area.height*step_int};

    ceres::Problem problem_inpaint;
    ceres::Solver::Summary summary;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
#ifdef VC_USE_CUDA_SPARSE
    // Check if Ceres was actually built with CUDA sparse support
    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::CUDA_SPARSE)) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;

        // Enable mixed precision for SPARSE_SCHUR
        if (options.linear_solver_type == ceres::SPARSE_SCHUR) {
            options.use_mixed_precision_solves = true;
        }
    } else {
        std::cerr << "Warning: CUDA_SPARSE requested but Ceres was not built with CUDA sparse support. Falling back to default solver." << std::endl;
    }
#endif
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 100;
    options.num_threads = omp_get_max_threads();
    options.use_nonmonotonic_steps = true;

    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++)
            if (state(j,i) & STATE_LOC_VALID) {
                data_new.surfs({j,i}).insert(&sm);
                data_new.loc(&sm, {j,i}) = {static_cast<double>(j),static_cast<double>(i)};
            }

    cv::Mat_<uint8_t> new_state = state.clone();

    //generate closed version of state
    cv::Mat m = cv::getStructuringElement(cv::MORPH_RECT, {3,3});

    uint8_t STATE_VALID = STATE_LOC_VALID | STATE_COORD_VALID;

    int res_count = 0;
    //slowly inpaint physics only points
    for(int r=0;r<closing_r+2;r++) {
        cv::Mat_<uint8_t> masked;
        bitwise_and(state, STATE_VALID, masked);
        cv::dilate(masked, masked, m, {-1,-1}, r);
        cv::erode(masked, masked, m, {-1,-1}, std::min(r,closing_r));
        // cv::imwrite("masked.tif", masked);

        for(int j=used_area.y;j<used_area.br().y;j++)
            for(int i=used_area.x;i<used_area.br().x;i++)
                if ((masked(j,i) & STATE_VALID) && (~new_state(j,i) & STATE_VALID)) {
                    new_state(j, i) = STATE_COORD_VALID;
                    points_new(j,i) = {-3,-2,-4};
                    //TODO add local area solve
                    double err = local_solve(&sm, {j,i}, data_new, new_state, points_new, step, src_step, LOSS_3D_INDIRECT | SURF_LOSS);
                    if (points_new(j,i)[0] == -3) {
                        //FIXME actually check for solver failure?
                        new_state(j, i) = 0;
                        points_new(j,i) = {-1,-1,-1};
                    }
                    else
                        res_count += surftrack_add_global(&sm, {j,i}, data_new, problem_inpaint, new_state, points_new, step*src_step, LOSS_3D_INDIRECT | OPTIMIZE_ALL);
                }
    }

    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++)
            if (state(j,i) & STATE_LOC_VALID)
                if (problem_inpaint.HasParameterBlock(&points_new(j,i)[0]))
                    problem_inpaint.SetParameterBlockConstant(&points_new(j,i)[0]);

    ceres::Solve(options, &problem_inpaint, &summary);
    std::cout << summary.BriefReport() << std::endl;

    cv::Mat_<cv::Vec3d> points_inpainted = points_new.clone();

    //TODO we could directly use higher res here?
    SurfaceMeta sm_inp;
    sm_inp._surf = new QuadSurface(points_inpainted, {1,1});

    SurfTrackerData data_inp;
    data_inp._data = data_new._data;

    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++)
            if (new_state(j,i) & STATE_LOC_VALID) {
                data_inp.surfs({j,i}).insert(&sm_inp);
                data_inp.loc(&sm_inp, {j,i}) = {static_cast<double>(j),static_cast<double>(i)};
            }

    ceres::Problem problem;

    std::cout << "optimizer: using " << used_area.tl() << used_area.br() << std::endl;

    int fix_points = 0;
    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++) {
            res_count += surftrack_add_global(&sm_inp, {j,i}, data_inp, problem, new_state, points_new, step*src_step, LOSS_3D_INDIRECT | SURF_LOSS | OPTIMIZE_ALL);
            fix_points++;
            if (problem.HasParameterBlock(&data_inp.loc(&sm_inp, {j,i})[0]))
                problem.AddResidualBlock(LinChkDistLoss::Create(data_inp.loc(&sm_inp, {j,i}), 1.0), nullptr, &data_inp.loc(&sm_inp, {j,i})[0]);
        }

    std::cout << "optimizer: num fix points " << fix_points << std::endl;

    data_inp.seed_loc = seed;
    data_inp.seed_coord = points_new(seed);

    int fix_points_z = 0;
    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++) {
            fix_points_z++;
            if (problem.HasParameterBlock(&data_inp.loc(&sm_inp, {j,i})[0]))
                problem.AddResidualBlock(ZLocationLoss<cv::Vec3d>::Create(points_new, data_inp.seed_coord[2] - (j-data.seed_loc[0])*step*src_step, z_loc_loss_w), new ceres::HuberLoss(1.0), &data_inp.loc(&sm_inp, {j,i})[0]);
        }

    std::cout << "optimizer: optimizing " << res_count << " residuals, seed " << seed << std::endl;

    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++)
            if (static_bounds.contains(cv::Point(i,j))) {
                if (problem.HasParameterBlock(&data_inp.loc(&sm_inp, {j,i})[0]))
                    problem.SetParameterBlockConstant(&data_inp.loc(&sm_inp, {j,i})[0]);
                if (problem.HasParameterBlock(&points_new(j, i)[0]))
                    problem.SetParameterBlockConstant(&points_new(j, i)[0]);
            }

    // additionally pin approved points anywhere in the active region -- kinda hacky because it does not allow the optimizer to work
    // as well with these points but if we know these points are good then we don't want them to move...
    if (pin_approved) {
        for (int j = used_area.y; j < used_area.br().y; ++j)
            for (int i = used_area.x; i < used_area.br().x; ++i)
                if (data.approvedAt({j,i})) {
                    if (problem.HasParameterBlock(&points_new(j,i)[0]))
                        problem.SetParameterBlockConstant(&points_new(j,i)[0]);
                    if (problem.HasParameterBlock(&data_inp.loc(&sm_inp, {j,i})[0]))
                        problem.SetParameterBlockConstant(&data_inp.loc(&sm_inp, {j,i})[0]);
                }
    }

    options.max_num_iterations = 1000;
    options.use_nonmonotonic_steps = true;
    options.use_inner_iterations = true;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
    std::cout << "optimizer: rms " << sqrt(summary.final_cost/summary.num_residual_blocks) << " count " << summary.num_residual_blocks << std::endl;

    {
        cv::Mat_<cv::Vec3f> points_hr_inp =
            surftrack_genpoints_hr(data, new_state, points_inpainted, used_area, step, src_step,
                                   /*inpaint=*/true, approved_weight_hr, prefer_approved_in_hr,
                                   /*parallel=*/hr_gen_parallel);
        try {
            auto dbg_surf = new QuadSurface(points_hr_inp(used_area_hr), {1/src_step,1/src_step});
            auto gen_channel = surftrack_generation_channel(generations, used_area, step);
            if (!gen_channel.empty())
                dbg_surf->setChannel("generations", gen_channel);
            std::string uuid = Z_DBG_GEN_PREFIX+get_surface_time_str()+"_inp_hr";
            dbg_surf->save(tgt_dir / uuid, uuid);
            delete dbg_surf;
        } catch (cv::Exception&) {
            // We did not find a valid region of interest to expand to
            std::cout << "optimizer: no valid region of interest found" << std::endl;
        }
    }

    cv::Mat_<cv::Vec3f> points_hr =
        surftrack_genpoints_hr(data, new_state, points_inpainted, used_area, step, src_step,
                               /*inpaint=*/false, approved_weight_hr, prefer_approved_in_hr,
                               /*parallel=*/hr_gen_parallel);
    SurfTrackerData data_out;
    cv::Mat_<cv::Vec3d> points_out(points.size(), {-1,-1,-1});
    cv::Mat_<uint8_t> state_out(state.size(), 0);
    cv::Mat_<uint8_t> support_count(state.size(), 0);

    // track if a remapped pixel had an approved surface in its LR neighborhood
    cv::Mat_<uint8_t> approved_near(state.size(), 0);

    std::cout << "remap: start used_area=" << used_area << " parallel=" << (remap_parallel?1:0) << std::endl;
#pragma omp parallel for if(remap_parallel)
    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++)
            if (static_bounds.contains(cv::Point(i,j))) {
                points_out(j, i) = points(j, i);
                state_out(j, i) = state(j, i);
                //FIXME copy surfs and locs
                mutex.lock();
                data_out.surfs({j,i}) = data.surfsC({j,i});
                for(auto &s : data_out.surfs({j,i}))
                    data_out.loc(s, {j,i}) = data.loc(s, {j,i});
                // copy approved pairs for static region
                for (auto &s : data_out.surfs({j,i}))
                    if (data.isApproved(s, {j,i}))
                        data_out.setApproved(s, {j,i});
                mutex.unlock();
            }
            else if (new_state(j,i) & STATE_VALID) {
                cv::Vec2d l = data_inp.loc(&sm_inp ,{j,i});
                int y = static_cast<int>(l[0]);
                int x = static_cast<int>(l[1]);
                l *= step;
                if (loc_valid(points_hr, l)) {
                    // Clamp HR interpolation location to ensure yi+1/xi+1 are in-bounds
                    l[0] = std::max(0.0, std::min<double>(l[0], points_hr.rows - 2 - 1e-6));
                    l[1] = std::max(0.0, std::min<double>(l[1], points_hr.cols - 2 - 1e-6));
                    // Clamp LR indices to ensure neighbor access (y+1,x+1) stays in-bounds
                    y = std::max(0, std::min(y, state.rows - 2));
                    x = std::max(0, std::min(x, state.cols - 2));
                    // mutex.unlock();
                    int src_loc_valid_count = 0;
                    if (state(y,x) & STATE_LOC_VALID)
                        src_loc_valid_count++;
                    if (state(y,x+1) & STATE_LOC_VALID)
                        src_loc_valid_count++;
                    if (state(y+1,x) & STATE_LOC_VALID)
                        src_loc_valid_count++;
                    if (state(y+1,x+1) & STATE_LOC_VALID)
                        src_loc_valid_count++;

                    support_count(j,i) = src_loc_valid_count;

                    points_out(j, i) = interp_lin_2d(points_hr, l);
                    state_out(j, i) = STATE_LOC_VALID | STATE_COORD_VALID;

                    // Build candidate surfaces from a configurable LR window around (y,x)
                    const int h = state.rows, w = state.cols;
                    const int rad = std::max(0, hr_attach_lr_radius);
                    std::set<SurfaceMeta*> surfs;
                    std::set<SurfaceMeta*> approved_neighborhood;
                    for (int oy = std::max(0, y - rad); oy <= std::min(h - 1, y + rad); ++oy) {
                        for (int ox = std::max(0, x - rad); ox <= std::min(w - 1, x + rad); ++ox) {
                            const auto& S = data.surfsC({oy, ox});
                            surfs.insert(S.begin(), S.end());
                            for (auto s : S) {
                                if (data.isApproved(s, {oy, ox}))
                                    approved_neighborhood.insert(s);
                            }
                        }
                    }

                    // remember if this output pixel is near any approved surface
                    if (!approved_neighborhood.empty())
                        approved_near(j,i) = 1;

                    // Relax acceptance for locally-approved surfaces (configurable)
                    const float approved_attach_relax = hr_attach_relax_factor;

                    for (auto& s : surfs) {
                        auto ptr = s->surface()->pointer();
                        const bool is_locally_approved = approved_neighborhood.contains(s);
                        const float thr = same_surface_th * (is_locally_approved ? approved_attach_relax : 1.0f);
                        // use relaxed thr also inside pointTo
                        float res = s->surface()->pointTo(ptr, points_out(j, i), thr, 10);
                        if (res <= thr) {
                            mutex.lock();
                            data_out.surfs({j,i}).insert(s);
                            cv::Vec3f loc = s->surface()->loc_raw(ptr);
                            data_out.loc(s, {j,i}) = {loc[1], loc[0]};
                            if (is_locally_approved)
                                data_out.setApproved(s, {j,i}); // propagate approval if neighborhood was approved
                            mutex.unlock();
                        }
                    }

                    // Fallback: if nothing attached but there is a locally-approved surface, attach the best one
                    if (data_out.surfs({j,i}).empty() && !approved_neighborhood.empty()) {
                        SurfaceMeta* best_s = nullptr;
                        float best_res = std::numeric_limits<float>::max();
                        cv::Vec3f best_loc_raw;
                        for (auto s : approved_neighborhood) {
                            auto ptr = s->surface()->pointer();
                            float thr = same_surface_th * approved_attach_relax;
                            float res = s->surface()->pointTo(ptr, points_out(j, i), thr, 10); //
                            if (res < best_res) {
                                best_res = res;
                                best_s = s;
                                best_loc_raw = s->surface()->loc_raw(ptr);
                            }
                        }
                        if (best_s && best_res <= same_surface_th * hr_attach_relax_factor) {
                            mutex.lock();
                            data_out.surfs({j,i}).insert(best_s);
                            data_out.loc(best_s, {j,i}) = {best_loc_raw[1], best_loc_raw[0]};
                            data_out.setApproved(best_s, {j,i});
                            mutex.unlock();
                        }
                    }
                }
            }
    std::cout << "remap: done" << std::endl;

    //now filter by consistency
    for(int j=used_area.y;j<used_area.br().y-1;j++)
        for(int i=used_area.x;i<used_area.br().x-1;i++)
            if (!static_bounds.contains(cv::Point(i,j)) && state_out(j,i) & STATE_VALID) {
                std::set<SurfaceMeta*> surf_src = data_out.surfs({j,i});
                for (auto s : surf_src) {
                    // never drop approved pairs if requested
                    if (keep_approved_on_consistency && data_out.isApproved(s, {j,i}))
                        continue;
                    int count;
                    float cost = local_cost(s, {j,i}, data_out, state_out, points_out, step, src_step, &count);
                    if (cost >= local_cost_inl_th /*|| count < 1*/) {
                        data_out.erase(s, {j,i});
                        data_out.eraseSurf(s, {j,i});
                    }
                }
            }

    cv::Mat_<uint8_t> fringe(state.size());
    cv::Mat_<uint8_t> fringe_next(state.size(), 1);
    int added = 1;
    for(int r=0;r<30 && added;r++) {
        ALifeTime timer("optimizer: add iteration\n");

        fringe_next.copyTo(fringe);
        fringe_next.setTo(0);

        added = 0;
#pragma omp parallel for collapse(2) schedule(dynamic)
        for(int j=used_area.y;j<used_area.br().y-1;j++)
            for(int i=used_area.x;i<used_area.br().x-1;i++)
                if (!static_bounds.contains(cv::Point(i,j)) && state_out(j,i) & STATE_LOC_VALID && (fringe(j, i) || fringe_next(j, i))) {
                    mutex.lock_shared();
                    std::set<SurfaceMeta*> surf_cands = data_out.surfs({j,i});
                    for(auto s : data_out.surfs({j,i}))
                        surf_cands.insert(s->overlapping.begin(), s->overlapping.end());
                    mutex.unlock_shared();

                    for(auto test_surf : surf_cands) {
                        mutex.lock_shared();
                        if (data_out.has(test_surf, {j,i})) {
                            mutex.unlock_shared();
                            continue;
                        }
                        mutex.unlock_shared();

                        auto ptr = test_surf->surface()->pointer();
                        if (test_surf->surface()->pointTo(ptr, points_out(j, i), same_surface_th, 10) > same_surface_th)
                            continue;

                        int count = 0;
                        cv::Vec3f loc_3d = test_surf->surface()->loc_raw(ptr);
                        int straight_count = 0;
                        float cost;
                        mutex.lock();
                        cost = local_cost_destructive(test_surf, {j,i}, data_out, state_out, points_out, step, src_step, loc_3d, &count, &straight_count);
                        mutex.unlock();

                        if (cost > local_cost_inl_th)
                            continue;

                        mutex.lock();
#pragma omp atomic
                        added++;
                        data_out.surfs({j,i}).insert(test_surf);
                        data_out.loc(test_surf, {j,i}) = {loc_3d[1], loc_3d[0]};
                        mutex.unlock();

                        for(int y=j-2;y<=j+2;y++)
                            for(int x=i-2;x<=i+2;x++)
                                if (y >= 0 && y < fringe_next.rows && x >= 0 && x < fringe_next.cols)
                                    fringe_next(y,x) = 1;
                    }
                }
        std::cout << "optimizer: added " << added << std::endl;
    }

    //reset unsupported points (keep if there is some LR support)
#pragma omp parallel for
    for(int j=used_area.y;j<used_area.br().y-1;j++)
        for(int i=used_area.x;i<used_area.br().x-1;i++)
            if (!static_bounds.contains(cv::Point(i,j))) {
                if (state_out(j,i) & STATE_LOC_VALID) {
                    // [APPROVED FIX] don't drop pixels in approved neighborhoods; keep geometry even if no surface attached
                    if (data_out.surfs({j,i}).empty() && support_count(j,i) == 0) {
                        if (approved_near(j,i)) {
                            state_out(j,i) = STATE_COORD_VALID;
                            // keep points_out(j,i) as-is
                        } else {
                            state_out(j,i) = 0;
                            points_out(j, i) = {-1,-1,-1};
                        }
                    }
                }
                else {
                    // no loc; only keep if approved neighborhood exists
                    if (approved_near(j,i)) {
                        state_out(j,i) = STATE_COORD_VALID;
                    } else {
                        state_out(j,i) = 0;
                        points_out(j, i) = {-1,-1,-1};
                    }
                }
            }

    points = points_out;
    state = state_out;
    data = data_out;
    data.seed_loc = seed;
    data.seed_coord = points(seed);

    for (int j = used_area.y; j < used_area.br().y; ++j)
        for (int i = used_area.x; i < used_area.br().x; ++i)
            if ((state(j,i) & STATE_LOC_VALID) == 0)
                generations(j,i) = 0;

    {
        cv::Mat_<cv::Vec3f> points_hr_inp =
            surftrack_genpoints_hr(data, state, points, used_area, step, src_step,
                                   /*inpaint=*/true, approved_weight_hr, prefer_approved_in_hr,
                                   /*parallel=*/hr_gen_parallel);
        try {
            auto dbg_surf = new QuadSurface(points_hr_inp(used_area_hr), {1/src_step,1/src_step});
            auto gen_channel = surftrack_generation_channel(generations, used_area, step);
            if (!gen_channel.empty())
                dbg_surf->setChannel("generations", gen_channel);
            std::string uuid = Z_DBG_GEN_PREFIX+get_surface_time_str()+"_opt_inp_hr";
            dbg_surf->save(tgt_dir / uuid, uuid);
            delete dbg_surf;
        } catch (cv::Exception&) {
            // We did not find a valid region of interest to expand to
            std::cout << "optimizer: no valid region of interest found" << std::endl;
        }
    }

    dbg_counter++;
}

QuadSurface *grow_surf_from_surfs(SurfaceMeta *seed, const std::vector<SurfaceMeta*> &surfs_v, const nlohmann::json &params, float voxelsize)
{
    bool flip_x = params.value("flip_x", 0);
    int global_steps_per_window = params.value("global_steps_per_window", 0);


    std::cout << "global_steps_per_window: " << global_steps_per_window << std::endl;
    std::cout << "flip_x: " << flip_x << std::endl;
    std::filesystem::path tgt_dir = params["tgt_dir"];

    std::unordered_map<std::string,SurfaceMeta*> surfs;
    float src_step = params.value("src_step", 20);
    float step = params.value("step", 10);
    const int step_int = surftrack_round_step(step);
    int max_width = params.value("max_width", 80000);

    local_cost_inl_th = params.value("local_cost_inl_th", 0.2f);
    same_surface_th = params.value("same_surface_th", 2.0f);
    straight_weight = params.value("straight_weight", 0.7f);            // Weight for 2D straight line constraints
    straight_weight_3D = params.value("straight_weight_3D", 4.0f);      // Weight for 3D straight line constraints
    sliding_w_scale = params.value("sliding_w_scale", 1.0f);            // Scale factor for sliding window
    z_loc_loss_w = params.value("z_loc_loss_w", 0.1f);                  // Weight for Z location loss constraints
    dist_loss_2d_w = params.value("dist_loss_2d_w", 1.0f);              // Weight for 2D distance constraints
    dist_loss_3d_w = params.value("dist_loss_3d_w", 2.0f);              // Weight for 3D distance constraints
    straight_min_count = params.value("straight_min_count", 1.0f);      // Minimum number of straight constraints
    inlier_base_threshold = params.value("inlier_base_threshold", 20);  // Starting threshold for inliers
    uint64_t deterministic_seed = uint64_t(params.value("deterministic_seed", 5489));
    double deterministic_jitter_px = params.value("deterministic_jitter_px", 0.15);

    // [APPROVED] knobs
    bool pin_approved_points = params.value("pin_approved_points", true);
    bool keep_approved_on_consistency = params.value("keep_approved_on_consistency", true);
    bool prefer_approved_in_hr = params.value("prefer_approved_in_hr", true);
    float approved_weight_hr = params.value("approved_weight_hr", 4.0f);

    // Optional hard z-range constraint: [z_min, z_max]
    bool enforce_z_range = false;
    double z_min = 0.0, z_max = 0.0;
    if (params.contains("z_range")) {
        try {
            if (params["z_range"].is_array() && params["z_range"].size() == 2) {
                z_min = params["z_range"][0].get<double>();
                z_max = params["z_range"][1].get<double>();
                if (z_min > z_max)
                    std::swap(z_min, z_max);
                enforce_z_range = true;
            }
        } catch (...) {
            // Ignore malformed z_range silently; fall back to no constraint
            enforce_z_range = false;
        }
    } else if (params.contains("z_min") && params.contains("z_max")) {
        try {
            z_min = params["z_min"].get<double>();
            z_max = params["z_max"].get<double>();
            if (z_min > z_max)
                std::swap(z_min, z_max);
            enforce_z_range = true;
        } catch (...) {
            enforce_z_range = false;
        }
    }

    std::cout << "  local_cost_inl_th: " << local_cost_inl_th << std::endl;
    std::cout << "  same_surface_th: " << same_surface_th << std::endl;
    std::cout << "  straight_weight: " << straight_weight << std::endl;
    std::cout << "  straight_weight_3D: " << straight_weight_3D << std::endl;
    std::cout << "  straight_min_count: " << straight_min_count << std::endl;
    std::cout << "  inlier_base_threshold: " << inlier_base_threshold << std::endl;
    std::cout << "  sliding_w_scale: " << sliding_w_scale << std::endl;
    std::cout << "  z_loc_loss_w: " << z_loc_loss_w << std::endl;
    std::cout << "  dist_loss_2d_w: " << dist_loss_2d_w << std::endl;
    std::cout << "  dist_loss_3d_w: " << dist_loss_3d_w << std::endl;
    std::cout << "  deterministic_seed: " << deterministic_seed << std::endl;
    std::cout << "  deterministic_jitter_px: " << deterministic_jitter_px << std::endl;
    if (enforce_z_range)
        std::cout << "  z_range: [" << z_min << ", " << z_max << "]" << std::endl;
    std::cout << "  pin_approved_points: " << pin_approved_points << std::endl;
    std::cout << "  keep_approved_on_consistency: " << keep_approved_on_consistency << std::endl;
    std::cout << "  prefer_approved_in_hr: " << prefer_approved_in_hr << std::endl;
    std::cout << "  approved_weight_hr: " << approved_weight_hr << std::endl;

    std::cout << "total surface count: " << surfs_v.size() << std::endl;

    std::set<SurfaceMeta*> approved_sm;

    std::set<std::string> used_approved_names;
    std::string log_filename = "/tmp/vc_grow_seg_from_segments_" + get_surface_time_str() + "_used_approved_segments.txt";
    std::ofstream approved_log(log_filename);

    for(auto &sm : surfs_v) {
        if (sm->meta->contains("tags") && sm->meta->at("tags").contains("approved"))
            approved_sm.insert(sm);
        if (!sm->meta->contains("tags") || !sm->meta->at("tags").contains("defective")) {
            surfs[sm->name()] = sm;
        }
    }

    for(auto sm : approved_sm)
        std::cout << "approved: " << sm->name() << std::endl;

    for(auto &sm : surfs_v)
        for(const auto& name : sm->overlapping_str)
            if (surfs.contains(name))
                sm->overlapping.insert(surfs[name]);

    std::cout << "total surface count (after defective filter): " << surfs.size() << std::endl;
    std::cout << "seed " << seed << " name " << seed->name() << " seed overlapping: "
              << seed->overlapping.size() << "/" << seed->overlapping_str.size() << std::endl;

    cv::Mat_<cv::Vec3f> seed_points = seed->surface()->rawPoints();

    int stop_gen = 100000;
    int closing_r = 20; //FIXME dont forget to reset!

    // Get sliding window scale from params (set earlier from JSON)

    //1k ~ 1cm, scaled by sliding_w_scale parameter
    int sliding_w = static_cast<int>(1000/src_step/step*2 * sliding_w_scale);
    int w = 2000/src_step/step*2+10+2*closing_r;
    int h = 15000/src_step/step*2+10+2*closing_r;
    cv::Size size = {w,h};
    cv::Rect bounds(0,0,w-1,h-1);
    cv::Rect save_bounds_inv(closing_r+5,closing_r+5,h-closing_r-10,w-closing_r-10);
    cv::Rect active_bounds(closing_r+5,closing_r+5,w-closing_r-10,h-closing_r-10);
    cv::Rect static_bounds(0,0,0,h);

    int x0 = w/2;
    int y0 = h/2;

    std::cout << "starting with size " << size << " seed " << cv::Vec2i(y0,x0) << std::endl;

    std::vector<cv::Vec2i> neighs = {{1,0},{0,1},{-1,0},{0,-1}};

    std::set<cv::Vec2i, Vec2iLess> fringe;

    cv::Mat_<uint8_t> state(size,0);
    cv::Mat_<uint16_t> generations(size, static_cast<uint16_t>(0));
    cv::Mat_<uint16_t> inliers_sum_dbg(size,0);
    cv::Mat_<cv::Vec3d> points(size,{-1,-1,-1});

    cv::Rect used_area(x0,y0,2,2);
    cv::Rect used_area_hr = {used_area.x*step_int, used_area.y*step_int, used_area.width*step_int, used_area.height*step_int};

    SurfTrackerData data;

    // Per-quad "already counted" mask for area accumulation (rows-1) x (cols-1)
    cv::Mat_<uint8_t> quad_done(cv::Size(std::max(0, w-1), std::max(0, h-1)), 0);
    // Live accumulated area in voxel^2 (progress logging). Updated O(1) per accepted vertex.
    double area_accum_vox2 = 0.0;
    auto init_area_scan = [&](void) {
        area_accum_vox2 = 0.0;
        quad_done = cv::Mat_<uint8_t>(cv::Size(std::max(0, points.cols-1), std::max(0, points.rows-1)), 0);
        for (int j = 0; j < state.rows - 1; ++j)
            for (int i = 0; i < state.cols - 1; ++i)
                area_accum_vox2 += maybe_quad_area_and_mark(j, i, state, points, quad_done);
    };

    cv::Vec2i seed_loc = {seed_points.rows/2, seed_points.cols/2};

    // Deterministic seed search around center using PRNG seeded by param
    {
        std::mt19937_64 rng(deterministic_seed);
        std::uniform_int_distribution<int> ry(0, seed_points.rows - 1);
        std::uniform_int_distribution<int> rx(0, seed_points.cols - 1);
        int tries = 0;
        while (seed_points(seed_loc)[0] == -1 ||
               (enforce_z_range && (seed_points(seed_loc)[2] < z_min || seed_points(seed_loc)[2] > z_max))) {
            seed_loc = {ry(rng), rx(rng)};
            if (++tries > 10000) break;
        }
        if (tries > 0) {
            std::cout << "deterministic seed search tries: " << tries << " got " << seed_loc << std::endl;
        }
    }

    data.loc(seed,{y0,x0}) = {static_cast<double>(seed_loc[0]), static_cast<double>(seed_loc[1])};
    data.surfs({y0,x0}).insert(seed);
    if (approved_sm.contains(seed)) data.setApproved(seed, {y0,x0});
    points(y0,x0) = data.lookup_int(seed,{y0,x0});


    data.seed_coord = points(y0,x0);
    data.seed_loc = cv::Point2i(y0,x0);

    std::cout << "seed coord " << data.seed_coord << " at " << data.seed_loc << std::endl;
    if (enforce_z_range && (data.seed_coord[2] < z_min || data.seed_coord[2] > z_max))
        std::cout << "warning: seed z " << data.seed_coord[2] << " is outside z_range; growth will be restricted to [" << z_min << ", " << z_max << "]" << std::endl;

    state(y0,x0) = STATE_LOC_VALID | STATE_COORD_VALID;
    generations(y0,x0) = 1;
    fringe.insert(cv::Vec2i(y0,x0));

    // Initial area is zero (need 2x2 block to form first quad)
    init_area_scan(); // harmless here; leaves accumulator at 0

    //insert initial surfs per location
    for(const auto& p : fringe) {
        data.surfs(p).insert(seed);
        if (approved_sm.contains(seed)) data.setApproved(seed, p);
        cv::Vec3f coord = points(p);
        std::cout << "testing " << p << " from cands: " << seed->overlapping.size() << coord << std::endl;
        for(auto s : seed->overlapping) {
            auto ptr = s->surface()->pointer();
            if (s->surface()->pointTo(ptr, coord, same_surface_th) <= same_surface_th) {
                cv::Vec3f loc = s->surface()->loc_raw(ptr);
                data.surfs(p).insert(s);
                data.loc(s, p) = {loc[1], loc[0]};
                if (approved_sm.contains(s)) data.setApproved(s, p);
            }
        }
        std::cout << "fringe point " << p << " surfcount " << data.surfs(p).size() << " init " << data.loc(seed, p) << data.lookup_int(seed, p) << std::endl;
    }

    std::cout << "starting from " << x0 << " " << y0 << std::endl;

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 200;

    int final_opts = global_steps_per_window;

    int loc_valid_count = 0;
    int succ = 0;
    int curr_best_inl_th = inlier_base_threshold;
    int last_succ_parametrization = 0;

    std::vector<SurfTrackerData> data_ths(omp_get_max_threads());
    std::vector<std::vector<cv::Vec2i>> added_points_threads(omp_get_max_threads());
    for(int i=0;i<omp_get_max_threads();i++)
        data_ths[i] = data;

    bool at_right_border = false;
    for(int generation=0;generation<stop_gen;generation++) {
        std::set<cv::Vec2i, Vec2iLess> cands;
        if (generation == 0) {
            cands.insert(cv::Vec2i(y0-1,x0));
        }
        else
            for(const auto& p : fringe)
            {
                if ((state(p) & STATE_LOC_VALID) == 0)
                    continue;

                for(const auto& n : neighs) {
                    cv::Vec2i pn = p+n;
                    if (save_bounds_inv.contains(cv::Point(pn))
                        && (state(pn) & STATE_PROCESSING) == 0
                        && (state(pn) & STATE_LOC_VALID) == 0)
                    {
                        state(pn) |= STATE_PROCESSING;
                        cands.insert(pn);
                    }
                    else if (!save_bounds_inv.contains(cv::Point(pn)) && save_bounds_inv.br().y <= pn[1]) {
                        at_right_border = true;
                    }
                }
            }
        fringe.clear();

        std::cout << "go with cands " << cands.size() << " inl_th " << curr_best_inl_th << std::endl;

        // Deterministic, sorted vector of candidates
        std::vector<cv::Vec2i> cands_vec(cands.begin(), cands.end());

        // Column-wise processing: grow left-to-right by column (x),
        // and within each column, process approved-nearby candidates first,
        // then the rest of the column's candidates.
        const int approved_priority_radius = params.value("approved_priority_radius", 2);
        struct Buckets { std::vector<cv::Vec2i> prio; std::vector<cv::Vec2i> other; };
        std::map<int, Buckets> by_col;
        for (const auto& p : cands_vec) {
            bool near_approved = false;
            for (int oy = std::max(0, p[0] - approved_priority_radius); oy <= std::min(h - 1, p[0] + approved_priority_radius) && !near_approved; ++oy) {
                for (int ox = std::max(0, p[1] - approved_priority_radius); ox <= std::min(w - 1, p[1] + approved_priority_radius); ++ox) {
                    if ((state(oy,ox) & STATE_LOC_VALID) && data.approvedAt({oy,ox})) { near_approved = true; break; }
                }
            }
            auto &bucket = by_col[p[1]];
            (near_approved ? bucket.prio : bucket.other).push_back(p);
        }

        std::shared_mutex mutex;
        int best_inliers_gen = 0;

        auto process_cands = [&](const std::vector<cv::Vec2i>& vec) {
#pragma omp parallel for schedule(static)
        for (int idx = 0; idx < static_cast<int>(vec.size()); ++idx) {
            int r = 1;
            cv::Vec2i p = vec[idx];

            if (state(p) & STATE_LOC_VALID)
                continue;

            if (points(p)[0] != -1)
                throw std::runtime_error("oops points(p)[0]");

            std::set<SurfaceMeta*> local_surfs = {seed};

            mutex.lock_shared();
            SurfTrackerData &data_th = data_ths[omp_get_thread_num()];
            int misses = 0;
            for(const auto& added : added_points_threads[omp_get_thread_num()]) {
                data_th.surfs(added) = data.surfs(added);
                for (auto &s : data.surfsC(added)) {
                    if (!data.has(s, added)) {
                        // Inconsistent: surface present without a stored mapping.
                        // Drop from thread-local set to avoid stale references.
                        data_th.surfs(added).erase(s);
                        misses++;
                        continue;
                    }
                    data_th.loc(s, added) = data.loc(s, added);
                    if (data.isApproved(s, added)) data_th.setApproved(s, added); // copy approvals to thread-local
                }
            }
            if (misses) {
                std::cout << "grow: cleaned " << misses << " stale surface refs in thread-local cache" << std::endl;
            }
            mutex.unlock_shared();
            mutex.lock();
            added_points_threads[omp_get_thread_num()].resize(0);
            mutex.unlock();

            for(int oy=std::max(p[0]-r,0);oy<=std::min(p[0]+r,h-1);oy++)
                for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,w-1);ox++)
                    if (state(oy,ox) & STATE_LOC_VALID) {
                        auto p_surfs = data_th.surfsC({oy,ox});
                        local_surfs.insert(p_surfs.begin(), p_surfs.end());
                    }
            // Also test all approved surfaces as "test surfaces"
            // but we won't use them as ref_surf (since they have no local locs here).
            std::set<SurfaceMeta*> test_surfs = local_surfs;
            if (params.value("consider_all_approved_as_candidates", true)) {
                test_surfs.insert(approved_sm.begin(), approved_sm.end());
            }

            cv::Vec3d best_coord = {-1,-1,-1};
            int best_inliers = -1;
            SurfaceMeta *best_surf = nullptr;
            cv::Vec2d best_loc = {-1,-1};
            bool best_ref_seed = false;
            bool best_approved = false;

            for(auto ref_surf : local_surfs) {
                int ref_count = 0;
                cv::Vec2d avg = {0,0};
                bool ref_seed = false;
                for(int oy=std::max(p[0]-r,0);oy<=std::min(p[0]+r,h-1);oy++)
                    for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,w-1);ox++)
                        if ((state(oy,ox) & STATE_LOC_VALID) && data_th.valid_int(ref_surf,{oy,ox})) {
                            ref_count++;
                            avg += data_th.loc(ref_surf,{oy,ox});
                            if (data_th.seed_loc == cv::Vec2i(oy,ox))
                                ref_seed = true;
                        }

                if (ref_count < 2 && !ref_seed)
                    continue;

                avg /= ref_count;

                // Deterministic symmetric jitter (tie-breaker) in [-jitter_px, +jitter_px),
                // salted by a stable key (surface name), not pointer value.
                uint64_t surf_key = mix64(uint64_t(std::hash<std::string>{}(ref_surf->name())));
                uint64_t salt = deterministic_seed ^ surf_key;
                double j0 = det_jitter_symm(p[0], p[1], salt) * deterministic_jitter_px;
                double j1 = det_jitter_symm(p[0], p[1], salt ^ 0x9e3779b97f4a7c15ULL) * deterministic_jitter_px;
                data_th.loc(ref_surf,p) = avg + cv::Vec2d(j0, j1);

                ceres::Problem problem;

                state(p) = STATE_LOC_VALID | STATE_COORD_VALID;

                int straight_count_init = 0;
                int count_init = surftrack_add_local(ref_surf, p, data_th, problem, state, points, step, src_step, LOSS_ZLOC, &straight_count_init);
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
                float cost_init = sqrt(summary.final_cost/count_init);

                bool fail = false;
                cv::Vec2d ref_loc = data_th.loc(ref_surf,p);

                if (!data_th.valid_int(ref_surf,p))
                    fail = true;

                cv::Vec3d coord;

                if (!fail) {
                    coord = data_th.lookup_int(ref_surf,p);
                    if (coord[0] == -1)
                        fail = true;
                }

                if (fail) {
                    data_th.erase(ref_surf, p);
                    continue;
                }

                state(p) = 0;
                generations(p) = 0;

                int inliers_sum = 0;
                int inliers_count = 0;

                //TODO could also have priorities!
                // Relaxable fast-track for approved surfaces via params
                int min_straight = params.value("approved_min_straight_in_grow", 0);
                int min_count    = params.value("approved_min_count_in_grow", 0);
                if (approved_sm.contains(ref_surf) &&
                    straight_count_init >= min_straight &&
                    count_init          >= min_count) {
                    // Respect z-range if enforced
                    if (enforce_z_range && (coord[2] < z_min || coord[2] > z_max)) {
                        data_th.erase(ref_surf, p);
                        continue;
                    }
                    std::cout << "found approved sm " << ref_surf->name() << std::endl;

                    // Log approved surface if not already logged
                    if (used_approved_names.insert(ref_surf->name()).second) {
                        mutex.lock();
                        approved_log << ref_surf->name() << std::endl;
                        approved_log.flush();
                        mutex.unlock();
                    }

                    best_inliers = 1000;
                    best_coord = coord;
                    best_surf = ref_surf;
                    best_loc = ref_loc;
                    best_ref_seed = ref_seed;
                    data_th.erase(ref_surf, p);
                    best_approved = true;
                    break;
                }

                for(auto test_surf : test_surfs) {
                    auto ptr = test_surf->surface()->pointer();
                    //FIXME this does not check geometry, only if its also on the surfaces (which might be good enough...)
                    if (test_surf->surface()->pointTo(ptr, coord, same_surface_th, 10) <= same_surface_th) {
                        int count = 0;
                        int straight_count = 0;
                        state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
                        cv::Vec3f loc = test_surf->surface()->loc_raw(ptr);
                        data_th.loc(test_surf, p) = {loc[1], loc[0]};
                        float cost = local_cost(test_surf, p, data_th, state, points, step, src_step, &count, &straight_count);
                        state(p) = 0;
                        generations(p) = 0;
                        data_th.erase(test_surf, p);
                        if (cost < local_cost_inl_th && (ref_seed || (count >= 2 && straight_count >= straight_min_count))) {
                            inliers_sum += count;
                            inliers_count++;
                        }
                    }
                }
                if ((inliers_count >= 2 || ref_seed) && inliers_sum > best_inliers) {
                    if (enforce_z_range && (coord[2] < z_min || coord[2] > z_max)) {
                        // Do not consider candidates outside the allowed z-range
                        data_th.erase(ref_surf, p);
                        continue;
                    }
                    best_inliers = inliers_sum;
                    best_coord = coord;
                    best_surf = ref_surf;
                    best_loc = ref_loc;
                    best_ref_seed = ref_seed;
                }
                data_th.erase(ref_surf, p);
            }

            if (points(p)[0] != -1)
                throw std::runtime_error("oops points(p)[0]");

            // Guard against duplicating existing 3D coords, even for approved shortcuts.
            // Previously this ran only for non-approved picks, which could result in
            // repeated identical quads when growing right with approved surfaces.
            if (best_inliers >= curr_best_inl_th || best_ref_seed)
            {
                if (enforce_z_range && (best_coord[2] < z_min || best_coord[2] > z_max)) {
                    // Final guard: reject best candidate outside z-range
                    best_inliers = -1;
                    best_ref_seed = false;
                } else if (used_area.width >=4 && used_area.height >= 4) {
                    cv::Vec2f tmp_loc_;
                    cv::Rect used_th = used_area;
                    float dist = pointTo(tmp_loc_, points(used_th), best_coord, same_surface_th, 1000, 1.0/(step*src_step));
                    tmp_loc_ += cv::Vec2f(used_th.x,used_th.y);
                    if (dist <= same_surface_th) {
                        int state_sum = state(tmp_loc_[1],tmp_loc_[0]) + state(tmp_loc_[1]+1,tmp_loc_[0]) + state(tmp_loc_[1],tmp_loc_[0]+1) + state(tmp_loc_[1]+1,tmp_loc_[0]+1);
                        best_inliers = -1;
                        best_ref_seed = false;
                        if (!state_sum)
                            throw std::runtime_error("this should not have any location?!");
                    }
                }
            }

            if (best_inliers >= curr_best_inl_th || best_ref_seed) {
                if (best_coord[0] == -1)
                    throw std::runtime_error("oops best_cord[0]");

                data_th.surfs(p).insert(best_surf);
                data_th.loc(best_surf, p) = best_loc;
                if (approved_sm.contains(best_surf)) data_th.setApproved(best_surf, p); // [APPROVED]
                state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
                points(p) = best_coord;
                inliers_sum_dbg(p) = best_inliers;
                generations(p) = static_cast<uint16_t>(std::min<int>(std::numeric_limits<uint16_t>::max(), generation + 1));

                ceres::Problem problem;
                surftrack_add_local(best_surf, p, data_th, problem, state, points, step, src_step, SURF_LOSS | LOSS_ZLOC);

                std::set<SurfaceMeta*> more_local_surfs;

                for(auto test_surf : test_surfs) {
                    for(auto s : test_surf->overlapping)
                        if (!local_surfs.contains(s) && s != best_surf)
                            more_local_surfs.insert(s);

                    if (test_surf == best_surf)
                        continue;

                    auto ptr = test_surf->surface()->pointer();
                    if (test_surf->surface()->pointTo(ptr, best_coord, same_surface_th, 10) <= same_surface_th) {
                        cv::Vec3f loc = test_surf->surface()->loc_raw(ptr);
                        cv::Vec3f coord = SurfTrackerData::lookup_int_loc(test_surf, {loc[1], loc[0]});
                        if (coord[0] == -1) {
                            continue;
                        }
                        int count = 0;
                        float cost = local_cost_destructive(test_surf, p, data_th, state, points, step, src_step, loc, &count);
                        if (cost < local_cost_inl_th) {
                            data_th.loc(test_surf, p) = {loc[1], loc[0]};
                            data_th.surfs(p).insert(test_surf);
                            if (approved_sm.contains(test_surf)) data_th.setApproved(test_surf, p); // [APPROVED]
                            surftrack_add_local(test_surf, p, data_th, problem, state, points, step, src_step, SURF_LOSS | LOSS_ZLOC);
                        }
                        else
                            data_th.erase(test_surf, p);
                    }
                }

                ceres::Solver::Summary summary;

                ceres::Solve(options, &problem, &summary);

                //TODO only add/test if we have 2 neighs which both find locations
                for(auto test_surf : test_surfs) {
                    auto ptr = test_surf->surface()->pointer();
                    float res = test_surf->surface()->pointTo(ptr, best_coord, same_surface_th, 10);
                    if (res <= same_surface_th) {
                        cv::Vec3f loc = test_surf->surface()->loc_raw(ptr);
                        cv::Vec3f coord = SurfTrackerData::lookup_int_loc(test_surf, {loc[1], loc[0]});
                        if (coord[0] == -1) {
                            continue;
                        }
                        int count = 0;
                        float cost = local_cost_destructive(test_surf, p, data_th, state, points, step, src_step, loc, &count);
                        if (cost < local_cost_inl_th) {
                            data_th.loc(test_surf, p) = {loc[1], loc[0]};
                            data_th.surfs(p).insert(test_surf);
                            if (approved_sm.contains(test_surf)) data_th.setApproved(test_surf, p); // [APPROVED]
                        };
                    }
                }

                mutex.lock();
                succ++;

                // Rebuild global set only from surfaces that have a valid loc in thread-local
                std::set<SurfaceMeta*> accepted;
                for (auto &s : data_th.surfs(p))
                    if (data_th.has(s, p))
                        accepted.insert(s);

                data.surfs(p).clear();
                for (auto &s : accepted) {
                    data.surfs(p).insert(s);
                    data.loc(s, p) = data_th.loc(s, p);
                }
                // rewrite approved flags for this location based on the accepted set
                data.clearApprovedAt(p);
                for (auto &s : accepted)
                    if (data_th.isApproved(s, p))
                        data.setApproved(s, p);

                for(int t=0;t<omp_get_max_threads();t++)
                    added_points_threads[t].push_back(p);

                if (!used_area.contains(cv::Point(p[1],p[0]))) {
                    used_area = used_area | cv::Rect(p[1],p[0],1,1);
                    used_area_hr = {used_area.x*step_int, used_area.y*step_int, used_area.width*step_int, used_area.height*step_int};
                }
                fringe.insert(p);
                // O(1) geometric area update: up to four quads become complete with a new vertex.
                area_accum_vox2 += maybe_quad_area_and_mark(p[0]-1, p[1]-1, state, points, quad_done);
                area_accum_vox2 += maybe_quad_area_and_mark(p[0]-1, p[1]  , state, points, quad_done);
                area_accum_vox2 += maybe_quad_area_and_mark(p[0]  , p[1]-1, state, points, quad_done);
                area_accum_vox2 += maybe_quad_area_and_mark(p[0]  , p[1]  , state, points, quad_done);
                mutex.unlock();
            }
            else if (best_inliers == -1) {
                //just try again some other time
                state(p) = 0;
                points(p) = {-1,-1,-1};
                generations(p) = 0;
            }
            else {
                state(p) = 0;
                points(p) = {-1,-1,-1};
                generations(p) = 0;
#pragma omp critical
                best_inliers_gen = std::max(best_inliers_gen, best_inliers);
            }
        }
        };

        // Process columns in ascending x. Within each, do approved-nearby first,
        // then the remaining in that column. This prevents moving past columns
        // with pending approved work.
        for (auto &kv : by_col) {
            process_cands(kv.second.prio);
            process_cands(kv.second.other);
        }

        if (generation == 1 && flip_x) {
            data.flip_x(x0);

            for(int i=0;i<omp_get_max_threads();i++) {
                data_ths[i] = data;
                added_points_threads[i].clear();
            }

            cv::Mat_<uint8_t> state_orig = state.clone();
            cv::Mat_<cv::Vec3d> points_orig = points.clone();
            cv::Mat_<uint16_t> generations_orig = generations.clone();
            state.setTo(0);
            points.setTo(cv::Vec3d(-1,-1,-1));
            generations.setTo(0);
            cv::Rect new_used_area = used_area;
            for(int j=used_area.y;j<=used_area.br().y+1;j++)
                for(int i=used_area.x;i<=used_area.br().x+1;i++)
                    if (state_orig(j, i)) {
                        int nx = x0+x0-i;
                        int ny = j;
                        state(ny, nx) = state_orig(j, i);
                        points(ny, nx) = points_orig(j, i);
                        generations(ny, nx) = generations_orig(j, i);
                        new_used_area = new_used_area | cv::Rect(nx,ny,1,1);
                    }

            used_area = new_used_area;
            used_area_hr = {used_area.x*step_int, used_area.y*step_int, used_area.width*step_int, used_area.height*step_int};

            fringe.clear();
            for(int j=used_area.y-2;j<=used_area.br().y+2;j++)
                for(int i=used_area.x-2;i<=used_area.br().x+2;i++)
                    if (state(j,i) & STATE_LOC_VALID)
                        fringe.insert(cv::Vec2i(j,i));
            // Geometry changed globally; rebuild area mask & accumulator
            init_area_scan();
        }

        int inl_lower_bound_reg = params.value("consensus_default_th", 10);
        int inl_lower_bound_b = params.value("consensus_limit_th", 2);
        int inl_lower_bound = inl_lower_bound_reg;

        if (!at_right_border && curr_best_inl_th <= inl_lower_bound)
            inl_lower_bound = inl_lower_bound_b;

        if (fringe.empty() && curr_best_inl_th > inl_lower_bound) {
            curr_best_inl_th -= (1+curr_best_inl_th-inl_lower_bound)/2;
            curr_best_inl_th = std::min(curr_best_inl_th, std::max(best_inliers_gen,inl_lower_bound));
            if (curr_best_inl_th >= inl_lower_bound) {
                cv::Rect active = active_bounds & used_area;
                for(int j=active.y-2;j<=active.br().y+2;j++)
                    for(int i=active.x-2;i<=active.br().x+2;i++)
                        if (state(j,i) & STATE_LOC_VALID)
                                fringe.insert(cv::Vec2i(j,i));
            }
        }
        else
            curr_best_inl_th = inlier_base_threshold;

        loc_valid_count = 0;
        for(int j=used_area.y;j<used_area.br().y-1;j++)
            for(int i=used_area.x;i<used_area.br().x-1;i++)
                if (state(j,i) & STATE_LOC_VALID)
                    loc_valid_count++;

        bool update_mapping = (succ >= 1000 && (loc_valid_count-last_succ_parametrization) >= std::max(100.0, 0.3*last_succ_parametrization));
        if (fringe.empty() && final_opts) {
            final_opts--;
            update_mapping = true;
        }

        if (!global_steps_per_window)
            update_mapping = false;

        if (generation % 50 == 0 || update_mapping /*|| generation < 10*/) {
            {
                cv::Mat_<cv::Vec3f> points_hr =
                    surftrack_genpoints_hr(data, state, points, used_area, step, src_step,
                                           /*inpaint=*/false, approved_weight_hr, prefer_approved_in_hr,
                                           /*parallel=*/params.value("hr_gen_parallel", false));
                auto dbg_surf = new QuadSurface(points_hr(used_area_hr), {1/src_step,1/src_step});
                dbg_surf->meta = new nlohmann::json;
                (*dbg_surf->meta)["vc_grow_seg_from_segments_params"] = params;

                auto gen_channel = surftrack_generation_channel(generations, used_area, step);
                if (!gen_channel.empty())
                    dbg_surf->setChannel("generations", gen_channel);

                (*dbg_surf->meta)["max_gen"] = surftrack_max_generation(generations, used_area);

                // Use exact geometric area accumulated so far
                const double area_exact_vx2 = area_accum_vox2;
                const double area_exact_cm2 = area_exact_vx2 * double(voxelsize) * double(voxelsize) / 1e8;
                (*dbg_surf->meta)["area_vx2"] = area_exact_vx2;
                (*dbg_surf->meta)["area_cm2"] = area_exact_cm2;
                (*dbg_surf->meta)["used_approved_segments"] = std::vector<std::string>(used_approved_names.begin(), used_approved_names.end());
                (*dbg_surf->meta)["seed_surface_name"] = seed->name();
                (*dbg_surf->meta)["seed_surface_id"] = seed->surface()->id;
                std::string uuid = Z_DBG_GEN_PREFIX+get_surface_time_str();
                dbg_surf->save(tgt_dir / uuid, uuid);
                delete dbg_surf;
            }
        }

        //lets just see what happens
        if (update_mapping) {
            dbg_counter = generation;
            SurfTrackerData opt_data = data;
            cv::Rect all(0,0,w, h);
            cv::Mat_<uint8_t> opt_state = state.clone();
            cv::Mat_<cv::Vec3d> opt_points = points.clone();
            cv::Mat_<uint16_t> opt_generations = generations.clone();

            cv::Rect active = active_bounds & used_area;
            optimize_surface_mapping(opt_data, opt_state, opt_points, opt_generations, active, static_bounds, step, src_step,
                                     {y0,x0}, closing_r, /*keep_inpainted=*/true, tgt_dir,
                                     /*pin_approved=*/pin_approved_points,
                                     /*approved_weight_hr=*/approved_weight_hr,
                                     /*prefer_approved_in_hr=*/prefer_approved_in_hr,
                                     /*keep_approved_on_consistency=*/keep_approved_on_consistency,
                                     /*hr_attach_lr_radius=*/params.value("hr_attach_lr_radius", 1),
                                     /*hr_attach_relax_factor=*/params.value("hr_attach_relax_factor", 2.0f),
                                     /*hr_gen_parallel=*/params.value("hr_gen_parallel", false),
                                     /*remap_parallel=*/params.value("remap_parallel", false));
            if (active.area() > 0) {
                copy(opt_data, data, active);
                opt_points(active).copyTo(points(active));
                opt_state(active).copyTo(state(active));
                opt_generations(active).copyTo(generations(active));

                for(int i=0;i<omp_get_max_threads();i++) {
                    data_ths[i] = data;
                    added_points_threads[i].resize(0);
                }
                // After remap, geometry/states may have changed -> recompute area mask/accumulator
                init_area_scan();
            }

            last_succ_parametrization = loc_valid_count;
            //recalc fringe after surface optimization (which often shrinks the surf)
            fringe.clear();
            curr_best_inl_th = inlier_base_threshold;
            for(int j=active.y-2;j<=active.br().y+2;j++)
                for(int i=active.x-2;i<=active.br().x+2;i++)
                    if (state(j,i) & STATE_LOC_VALID)
                        fringe.insert(cv::Vec2i(j,i));

            {
                cv::Mat_<cv::Vec3f> points_hr =
                    surftrack_genpoints_hr(data, state, points, used_area, step, src_step,
                                           /*inpaint=*/false, approved_weight_hr, prefer_approved_in_hr,
                                           /*parallel=*/params.value("hr_gen_parallel", false));
                auto dbg_surf = new QuadSurface(points_hr(used_area_hr), {1/src_step,1/src_step});
                dbg_surf->meta = new nlohmann::json;
                (*dbg_surf->meta)["vc_grow_seg_from_segments_params"] = params;

                auto gen_channel = surftrack_generation_channel(generations, used_area, step);
                if (!gen_channel.empty())
                    dbg_surf->setChannel("generations", gen_channel);

                (*dbg_surf->meta)["max_gen"] = surftrack_max_generation(generations, used_area);

                std::string uuid = Z_DBG_GEN_PREFIX+get_surface_time_str()+"_opt";
                const double area_exact_vx2 = area_accum_vox2;
                const double area_exact_cm2 = area_exact_vx2 * double(voxelsize) * double(voxelsize) / 1e8;
                (*dbg_surf->meta)["area_vx2"] = area_exact_vx2;
                (*dbg_surf->meta)["area_cm2"] = area_exact_cm2;
                (*dbg_surf->meta)["used_approved_segments"] = std::vector<std::string>(used_approved_names.begin(), used_approved_names.end());
                (*dbg_surf->meta)["seed_surface_name"] = seed->name();
                (*dbg_surf->meta)["seed_surface_id"] = seed->surface()->id;
                dbg_surf->save(tgt_dir / uuid, uuid);
                delete dbg_surf;
            }
        }

        const double current_area_vx2 = area_accum_vox2;
        const double current_area_cm2 = current_area_vx2 * double(voxelsize) * double(voxelsize) / 1e8;
        printf("gen %d processing %lu fringe cands (total done %d fringe: %lu) area %.0f vx^2 (%f cm^2) best th: %d\n",
               generation, static_cast<unsigned long>(cands.size()), succ, static_cast<unsigned long>(fringe.size()),
               current_area_vx2, current_area_cm2, best_inliers_gen);

        //continue expansion
        if (fringe.empty() && w < max_width/step)
        {
            at_right_border = false;
            std::cout << "expanding by " << sliding_w << std::endl;

            std::cout << size << bounds << save_bounds_inv << used_area << active_bounds << (used_area & active_bounds) << static_bounds << std::endl;
            final_opts = global_steps_per_window;
            w += sliding_w;
            size = {w,h};
            bounds = {0,0,w-1,h-1};
            save_bounds_inv = {closing_r+5,closing_r+5,h-closing_r-10,w-closing_r-10};

            cv::Mat_<cv::Vec3d> old_points = points;
            points = cv::Mat_<cv::Vec3d>(size, {-1,-1,-1});
            old_points.copyTo(points(cv::Rect(0,0,old_points.cols,h)));

            cv::Mat_<uint8_t> old_state = state;
            state = cv::Mat_<uint8_t>(size, 0);
            old_state.copyTo(state(cv::Rect(0,0,old_state.cols,h)));

            cv::Mat_<uint16_t> old_generations = generations;
            generations = cv::Mat_<uint16_t>(size, static_cast<uint16_t>(0));
            old_generations.copyTo(generations(cv::Rect(0,0,old_generations.cols,h)));

            cv::Mat_<uint16_t> old_inliers_sum_dbg = inliers_sum_dbg;
            inliers_sum_dbg = cv::Mat_<uint16_t>(size, 0);
            old_inliers_sum_dbg.copyTo(inliers_sum_dbg(cv::Rect(0,0,old_inliers_sum_dbg.cols,h)));

            int overlap = 5;
            active_bounds = {w-sliding_w-2*closing_r-10-overlap,closing_r+5,sliding_w+2*closing_r+10+overlap,h-closing_r-10};
            static_bounds = {0,0,w-sliding_w-2*closing_r-10,h};

            cv::Rect active = active_bounds & used_area;

            std::cout << size << bounds << save_bounds_inv << used_area << active_bounds << (used_area & active_bounds) << static_bounds << std::endl;
            fringe.clear();
            curr_best_inl_th = inlier_base_threshold;
            for(int j=active.y-2;j<=active.br().y+2;j++)
                for(int i=active.x-2;i<=active.br().x+2;i++)
                    //FIXME why isn't this working?!'
                    if (state(j,i) & STATE_LOC_VALID)
                        fringe.insert(cv::Vec2i(j,i));
            // Grid grew horizontally; rebuild area mask/accumulator in new shape
            init_area_scan();
        }

        cv::imwrite(tgt_dir / "inliers_sum.tif", inliers_sum_dbg(used_area), surftrack_tiff_compression_params());

        if (fringe.empty())
            break;
    }

    approved_log.close();

    // Final exact surface area from final optimized geometry.
    double area_final_vox2 = 0.0;
    for (int j = 0; j < state.rows - 1; ++j)
        for (int i = 0; i < state.cols - 1; ++i)
            if ( (state(j,   i  ) & STATE_LOC_VALID) &&
                 (state(j,   i+1) & STATE_LOC_VALID) &&
                 (state(j+1, i  ) & STATE_LOC_VALID) &&
                 (state(j+1, i+1) & STATE_LOC_VALID) )
            {
                area_final_vox2 += vc::surface::quadAreaVox2(points(j,   i  ),
                                                            points(j,   i+1),
                                                            points(j+1, i  ),
                                                            points(j+1, i+1));
            }
    const double area_final_cm2 = area_final_vox2 * double(voxelsize) * double(voxelsize) / 1e8;
    std::cout << "area exact: " << area_final_vox2 << " vx^2 (" << area_final_cm2 << " cm^2)" << std::endl;

    cv::Mat_<cv::Vec3f> points_hr =
        surftrack_genpoints_hr(data, state, points, used_area, step, src_step,
                               /*inpaint=*/false, approved_weight_hr, prefer_approved_in_hr);

    auto surf = new QuadSurface(points_hr(used_area_hr), {1/src_step,1/src_step});

    auto gen_channel = surftrack_generation_channel(generations, used_area, step);
    if (!gen_channel.empty())
        surf->setChannel("generations", gen_channel);

    surf->meta = new nlohmann::json;
    (*surf->meta)["max_gen"] = surftrack_max_generation(generations, used_area);
    (*surf->meta)["area_vx2"] = area_final_vox2;
    (*surf->meta)["area_cm2"] = area_final_cm2;
    (*surf->meta)["used_approved_segments"] = std::vector<std::string>(used_approved_names.begin(), used_approved_names.end());
    (*surf->meta)["seed_surface_name"] = seed->name();
    (*surf->meta)["seed_surface_id"] = seed->surface()->id;

    return surf;
}
