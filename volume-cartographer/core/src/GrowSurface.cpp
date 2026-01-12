#include <omp.h>
#include <random>

#include <opencv2/imgcodecs.hpp>
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
#include "vc/core/util/SurfaceModeling.hpp"
#include "vc/core/util/SurfaceArea.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/core/util/Tiff.hpp"
#include "vc/core/types/ChunkedTensor.hpp"

#include "vc/core/util/LifeTime.hpp"

#include <atomic>
#include <chrono>
#include <array>
#include <fstream>
#include <iostream>
#include <shared_mutex>
#include <mutex>
#include <memory>
#include <filesystem>
#include <functional>
#include <limits>
#include <algorithm>
#include <map>
#include <unordered_set>
#include <cmath>
#include <vector>


int static dbg_counter = 0;
// Default values for thresholds Will be configurable through JSON
static float local_cost_inl_th = 0.2;
static float assoc_surface_th = 2.0f;        // Threshold for surface association
static float duplicate_surface_th = 2.0f;    // Threshold for duplicate rejection
static float remap_attach_surface_th = 2.0f; // Threshold for remap attachment
static int point_to_max_iters = 10;          // Max iterations for pointTo searches
static float straight_weight = 0.7f;       // Weight for 2D straight line constraints
static float straight_weight_3D = 4.0f;    // Weight for 3D straight line constraints
static float sliding_w_scale = 1.0f;       // Scale factor for sliding window
static float z_loc_loss_w = 0.1f;          // Weight for Z location loss constraints
static float dist_loss_2d_w = 1.0f;        // Weight for 2D distance constraints
static float dist_loss_3d_w = 2.0f;        // Weight for 3D distance constraints
static float dist_loss_xy_w = 1.0f;        // Weight for XY distance constraints (same-row)
static float normal_loss_w = 1.0f;         // Weight for normal grid alignment loss
static float snap_loss_w = 0.1f;           // Weight for normal grid snapping loss
static float straight_min_count = 1.0f;    // Minimum number of straight constraints
static int inlier_base_threshold = 20;     // Starting threshold for inliers
static int heading_window = 9;             // Window (cols/rows) to estimate heading
static int heading_min_segments = 3;       // Minimum segments required to apply heading loss
static float heading_loss_w = 0.5f;        // Weight for heading loss (XY plane)
static bool seed_pointto_from_neighbors = false; // Seed pointTo from neighbor locs (disabled by default)
static int normal_grid_z_min = -1;
static int normal_grid_z_max = std::numeric_limits<int>::max();
static std::unique_ptr<vc::core::util::NormalGridVolume> normal_grid_volume;

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

template <typename T>
static inline bool point_in_bounds(const cv::Mat_<T>& mat, const cv::Vec2i& p)
{
    return p[0] >= 0 && p[1] >= 0 && p[0] < mat.rows && p[1] < mat.cols;
}

using SurfPoint = std::pair<QuadSurface*,cv::Vec2i>;

struct HeadingExpectation {
    cv::Vec2d dir;
    double weight = 0.0;
    int segments = 0;
};

struct HeadingLossXY {
    HeadingLossXY(const cv::Vec2d& dir, double w)
        : _dir(dir), _w(w) {}
    template <typename T>
    bool operator()(const T* const a, const T* const b, T* residual) const {
        T dx = b[0] - a[0];
        T dy = b[1] - a[1];
        T len = sqrt(dx*dx + dy*dy);
        if (len <= T(0)) {
            residual[0] = T(0);
            return true;
        }

        T dot = (dx * T(_dir[0]) + dy * T(_dir[1])) / len;
        residual[0] = T(_w) * (T(1) - dot);
        return true;
    }

    cv::Vec2d _dir;
    double _w;

    static ceres::CostFunction* Create(const cv::Vec2d& dir, double w)
    {
        return new ceres::AutoDiffCostFunction<HeadingLossXY, 1, 3, 3>(
            new HeadingLossXY(dir, w));
    }
};

static inline bool point_valid_heading(const cv::Mat_<cv::Vec3d>& points,
                                       const cv::Mat_<uint8_t>& state,
                                       const cv::Vec2i& p)
{
    if (!point_in_bounds(state, p))
        return false;
    if ((state(p) & STATE_LOC_VALID) == 0)
        return false;
    return points(p)[0] != -1;
}

static bool compute_heading_expectation_xy(const cv::Mat_<cv::Vec3d>& points,
                                           const cv::Mat_<uint8_t>& state,
                                           const cv::Vec2i& p,
                                           const cv::Vec2i& step_dir,
                                           int window,
                                           HeadingExpectation* out)
{
    std::vector<cv::Vec2d> dirs;
    dirs.reserve(window);

    for (int k = 1; k <= window; ++k) {
        cv::Vec2i a = {p[0] + step_dir[0] * k, p[1] + step_dir[1] * k};
        cv::Vec2i b = {p[0] + step_dir[0] * (k + 1), p[1] + step_dir[1] * (k + 1)};
        if (!point_valid_heading(points, state, a) || !point_valid_heading(points, state, b))
            break;

        const cv::Vec3d& pa = points(a);
        const cv::Vec3d& pb = points(b);
        cv::Vec2d d = {pa[0] - pb[0], pa[1] - pb[1]};
        double len = cv::norm(d);
        if (len < 1e-6)
            continue;
        dirs.push_back(d / len);
    }

    if (dirs.size() < static_cast<size_t>(heading_min_segments))
        return false;

    cv::Vec2d sum(0.0, 0.0);
    for (const auto& d : dirs)
        sum += d;
    double sum_len = cv::norm(sum);
    if (sum_len < 1e-6)
        return false;

    out->dir = sum / sum_len;
    out->weight = sum_len / static_cast<double>(dirs.size());
    out->segments = static_cast<int>(dirs.size());
    return true;
}

static int add_heading_loss_axis(cv::Mat_<cv::Vec3d>& points,
                                 const cv::Mat_<uint8_t>& state,
                                 const cv::Vec2i& p,
                                 const cv::Vec2i& dir_a,
                                 const cv::Vec2i& dir_b,
                                 ceres::Problem& problem,
                                 int flags)
{
    HeadingExpectation exp_a;
    HeadingExpectation exp_b;
    bool has_a = compute_heading_expectation_xy(points, state, p, dir_a, heading_window, &exp_a);
    bool has_b = compute_heading_expectation_xy(points, state, p, dir_b, heading_window, &exp_b);

    if (!has_a && !has_b)
        return 0;

    const HeadingExpectation* best = &exp_a;
    cv::Vec2i best_dir = dir_a;
    if (!has_a || (has_b && exp_b.segments > exp_a.segments)) {
        best = &exp_b;
        best_dir = dir_b;
    }

    cv::Vec2i prev = {p[0] + best_dir[0], p[1] + best_dir[1]};
    if (!point_valid_heading(points, state, p) || !point_valid_heading(points, state, prev))
        return 0;

    const double weight = heading_loss_w * best->weight;
    if (weight <= 0.0)
        return 0;

    problem.AddResidualBlock(HeadingLossXY::Create(best->dir, weight), nullptr,
                             &points(prev)[0], &points(p)[0]);
    if ((flags & OPTIMIZE_ALL) == 0)
        problem.SetParameterBlockConstant(&points(prev)[0]);

    return 1;
}

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

struct SurfPtrLess {
    bool operator()(const QuadSurface* a, const QuadSurface* b) const {
        return std::less<const QuadSurface*>()(a, b);
    }
};

class SurfPtrSet {
public:
    using value_type = QuadSurface*;
    using container_type = std::vector<value_type>;
    using iterator = container_type::iterator;
    using const_iterator = container_type::const_iterator;

    SurfPtrSet() = default;

    template <typename It>
    SurfPtrSet(It first, It last) {
        insert(first, last);
    }

    bool insert(value_type value) {
        auto it = std::lower_bound(_items.begin(), _items.end(), value, SurfPtrLess{});
        if (it != _items.end() && *it == value)
            return false;
        _items.insert(it, value);
        return true;
    }

    template <typename It>
    void insert(It first, It last) {
        for (; first != last; ++first)
            insert(*first);
    }

    void erase(value_type value) {
        auto it = std::lower_bound(_items.begin(), _items.end(), value, SurfPtrLess{});
        if (it != _items.end() && *it == value)
            _items.erase(it);
    }

    bool contains(value_type value) const {
        return std::binary_search(_items.begin(), _items.end(), value, SurfPtrLess{});
    }

    void clear() {
        _items.clear();
    }

    bool empty() const {
        return _items.empty();
    }

    size_t size() const {
        return _items.size();
    }

    iterator begin() {
        return _items.begin();
    }

    iterator end() {
        return _items.end();
    }

    const_iterator begin() const {
        return _items.begin();
    }

    const_iterator end() const {
        return _items.end();
    }

private:
    container_type _items;
};

class resId_t
{
public:
    resId_t() : _type(0), _sm(nullptr) {
    } ;
    resId_t(int type, QuadSurface* sm, const cv::Vec2i& p) : _type(type), _sm(sm), _p(p) {};
    resId_t(int type, QuadSurface* sm, const cv::Vec2i &a, const cv::Vec2i &b) : _type(type), _sm(sm)
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
    QuadSurface* _sm;
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
    cv::Vec2d &loc(QuadSurface *sm, const cv::Vec2i &loc)
    {
        return _data[{sm,loc}];
    }
    bool getLoc(QuadSurface *sm, const cv::Vec2i &loc, cv::Vec2d *out) const
    {
        auto it = _data.find({sm,loc});
        if (it == _data.end())
            return false;
        *out = it->second;
        return true;
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
    bool has(QuadSurface *sm, const cv::Vec2i &loc) const {
        return _data.contains({sm,loc});
    }
    void erase(QuadSurface *sm, const cv::Vec2i &loc)
    {
        _data.erase({sm,loc});
    }
    void eraseSurf(QuadSurface *sm, const cv::Vec2i &loc)
    {
        _surfs[loc].erase(sm);
    }
    SurfPtrSet &surfs(const cv::Vec2i &loc)
    {
        return _surfs[loc];
    }
    const SurfPtrSet &surfsC(const cv::Vec2i &loc) const
    {
        if (!_surfs.contains(loc))
            return _emptysurfs;
        else
            return _surfs.find(loc)->second;
    }
    cv::Vec3d lookup_int(QuadSurface *sm, const cv::Vec2i &p)
    {
        auto id = std::make_pair(sm,p);
        if (!_data.contains(id))
            throw std::runtime_error("error, lookup failed!");
        cv::Vec2d l = loc(sm, p);
        if (!loc_valid(sm->rawPoints(), l))
            return {-1,-1,-1};
        return at_int_inv(sm->rawPoints(), l);
    }
    bool valid_int(QuadSurface *sm, const cv::Vec2i &p)
    {
        auto id = std::make_pair(sm,p);
        if (!_data.contains(id))
            return false;
        cv::Vec2d l = loc(sm, p);
        if (l[0] == -1)
            return false;
        else {
            cv::Rect bounds = {0, 0, sm->rawPoints().rows-2,sm->rawPoints().cols-2};
            cv::Vec2i li = {static_cast<int>(floor(l[0])),static_cast<int>(floor(l[1]))};
            if (bounds.contains(cv::Point(li)))
            {
                if (sm->rawPoints()(li[0],li[1])[0] == -1)
                    return false;
                if (sm->rawPoints()(li[0]+1,li[1])[0] == -1)
                    return false;
                if (sm->rawPoints()(li[0],li[1]+1)[0] == -1)
                    return false;
                if (sm->rawPoints()(li[0]+1,li[1]+1)[0] == -1)
                    return false;
                return true;
            }
            else
                return false;
        }
    }
    static cv::Vec3d lookup_int_loc(QuadSurface *sm, const cv::Vec2f &l)
    {
        if (!loc_valid(sm->rawPoints(), cv::Vec2d(l[0], l[1])))
            return {-1,-1,-1};
        return at_int_inv(sm->rawPoints(), l);
    }
    void flip_x(int x0)
    {
        std::cout << " src sizes " << _data.size() << " " << _surfs.size() << std::endl;
        SurfTrackerData old = *this;
        _data.clear();
        _res_blocks.clear();
        _surfs.clear();

        for(auto &it : old._data)
            _data[{it.first.first,{it.first.second[0],x0+x0-it.first.second[1]}}] = it.second;

        for(auto &it : old._surfs)
            _surfs[{it.first[0],x0+x0-it.first[1]}] = it.second;

        std::cout << " flipped sizes " << _data.size() << " " << _surfs.size() << std::endl;
    }

// protected:
    std::unordered_map<SurfPoint,cv::Vec2d,SurfPoint_hash> _data;
    std::unordered_map<resId_t,ceres::ResidualBlockId,resId_hash> _res_blocks;
    std::unordered_map<cv::Vec2i, SurfPtrSet> _surfs;
    SurfPtrSet _emptysurfs;
    cv::Vec3d seed_coord;
    cv::Vec2i seed_loc;
};

static int add_normal_grid_loss(ceres::Problem &problem,
                                const cv::Vec2i &p,
                                const cv::Mat_<uint8_t> &state,
                                cv::Mat_<cv::Vec3d> &points,
                                int flags = 0)
{
    if (!normal_grid_volume)
        return 0;
    if (normal_loss_w <= 0.0f && snap_loss_w <= 0.0f)
        return 0;

    cv::Vec2i p_br = p + cv::Vec2i(1,1);
    if (!point_in_bounds(state, p) || !point_in_bounds(state, p_br))
        return 0;

    auto valid_state = [&](const cv::Vec2i& pt) {
        return (state(pt) & (STATE_COORD_VALID | STATE_LOC_VALID)) != 0;
    };
    if (!valid_state(p) || !valid_state({p[0], p[1] + 1}) ||
        !valid_state({p[0] + 1, p[1]}) || !valid_state(p_br)) {
        return 0;
    }

    cv::Vec2i p_tr = {p[0], p[1] + 1};
    cv::Vec2i p_bl = {p[0] + 1, p[1]};

    double* pA = &points(p)[0];
    double* pB1 = &points(p_tr)[0];
    double* pB2 = &points(p_bl)[0];
    double* pC = &points(p_br)[0];

    int count = 0;
    for (int i = 0; i < 3; ++i) {
        const bool direction_aware = false;
        problem.AddResidualBlock(
            NormalConstraintPlane::Create(*normal_grid_volume, i, normal_loss_w, snap_loss_w,
                                          direction_aware, normal_grid_z_min, normal_grid_z_max),
            nullptr, pA, pB1, pB2, pC);
        problem.AddResidualBlock(
            NormalConstraintPlane::Create(*normal_grid_volume, i, normal_loss_w, snap_loss_w,
                                          direction_aware, normal_grid_z_min, normal_grid_z_max),
            nullptr, pC, pB2, pB1, pA);
        problem.AddResidualBlock(
            NormalConstraintPlane::Create(*normal_grid_volume, i, normal_loss_w, snap_loss_w,
                                          direction_aware, normal_grid_z_min, normal_grid_z_max),
            nullptr, pB1, pC, pA, pB2);
        problem.AddResidualBlock(
            NormalConstraintPlane::Create(*normal_grid_volume, i, normal_loss_w, snap_loss_w,
                                          direction_aware, normal_grid_z_min, normal_grid_z_max),
            nullptr, pB2, pA, pC, pB1);
        count += 4;
    }

    if ((flags & OPTIMIZE_ALL) == 0) {
        problem.SetParameterBlockConstant(&points(p_tr)[0]);
        problem.SetParameterBlockConstant(&points(p_bl)[0]);
        problem.SetParameterBlockConstant(&points(p_br)[0]);
    }

    return count;
}

static bool seed_ptr_from_neighbors(QuadSurface* surf,
                                    SurfTrackerData& data,
                                    const cv::Mat_<uint8_t>& state,
                                    const cv::Vec2i& p,
                                    const cv::Vec3f& target,
                                    cv::Vec3f* ptr_out)
{
    static const std::array<cv::Vec2i, 8> kDirs = {
        cv::Vec2i(0, -1), cv::Vec2i(0, 1), cv::Vec2i(-1, 0), cv::Vec2i(1, 0),
        cv::Vec2i(-1, -1), cv::Vec2i(-1, 1), cv::Vec2i(1, -1), cv::Vec2i(1, 1)
    };

    const auto* surf_points = surf->rawPointsPtr();
    if (!surf_points)
        return false;

    double best_dist = std::numeric_limits<double>::infinity();
    cv::Vec2d best_loc;
    bool found = false;

    for (const auto& dir : kDirs) {
        const cv::Vec2i n = p + dir;
        if (!point_in_bounds(state, n))
            continue;
        if ((state(n) & STATE_LOC_VALID) == 0)
            continue;

        cv::Vec2d loc;
        if (!data.getLoc(surf, n, &loc))
            continue;
        if (!loc_valid(*surf_points, loc))
            continue;

        const cv::Vec3f coord = at_int_inv(*surf_points,
                                           {static_cast<float>(loc[0]), static_cast<float>(loc[1])});
        if (coord[0] == -1)
            continue;

        const double dist = cv::norm(cv::Vec3d(coord) - cv::Vec3d(target));
        if (dist < best_dist) {
            best_dist = dist;
            best_loc = loc;
            found = true;
        }
    }

    if (!found)
        return false;

    const cv::Vec3f center = surf->center();
    const cv::Vec2f scale = surf->scale();
    *ptr_out = cv::Vec3f(static_cast<float>(best_loc[1] - center[0] * scale[0]),
                         static_cast<float>(best_loc[0] - center[1] * scale[1]),
                         0.0f);
    return true;
}

static float pointTo_seeded_neighbor(QuadSurface* surf,
                                     SurfTrackerData& data,
                                     const cv::Mat_<uint8_t>& state,
                                     const cv::Vec2i& p,
                                     const cv::Vec3f& target,
                                     float th,
                                     int max_iters,
                                     SurfacePatchIndex* surface_patch_index,
                                     cv::Vec3f* ptr_out)
{
    cv::Vec3f ptr;
    if (!seed_pointto_from_neighbors ||
        !seed_ptr_from_neighbors(surf, data, state, p, target, &ptr)) {
        ptr = surf->pointer();
    }

    const float res = surf->pointTo(ptr, target, th, max_iters, surface_patch_index);
    if (ptr_out)
        *ptr_out = ptr;
    return res;
}

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

    for(auto &it : src._data)
        if (roi.contains(cv::Point(it.first.second)))
            tgt._data[it.first] = it.second;
    for(auto &it : src._surfs)
        if (roi.contains(cv::Point(it.first)))
            tgt._surfs[it.first] = it.second;

    // tgt.seed_loc = src.seed_loc;
    // tgt.seed_coord = src.seed_coord;
}

static int add_surftrack_distloss(QuadSurface *sm, const cv::Vec2i &p, const cv::Vec2i &off, SurfTrackerData &data,
    ceres::Problem &problem, const cv::Mat_<uint8_t> &state, float unit, int flags = 0, ceres::ResidualBlockId *res = nullptr, float w = 1.0)
{
    if (!point_in_bounds(state, p) || !point_in_bounds(state, p + off))
        return 0;
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
    if (!point_in_bounds(state, p) || !point_in_bounds(state, p + off))
        return 0;
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

// XY-plane distance loss for same-row adjacent points
// Complements z_loc_loss: ensures XY displacement is maintained along rows
static int add_surftrack_distloss_XY(cv::Mat_<cv::Vec3d> &points, const cv::Vec2i &p, const cv::Vec2i &off,
    ceres::Problem &problem, const cv::Mat_<uint8_t> &state, float unit, int flags = 0)
{
    if (!point_in_bounds(state, p) || !point_in_bounds(state, p + off))
        return 0;
    if ((state(p) & (STATE_COORD_VALID | STATE_LOC_VALID)) == 0)
        return 0;
    if ((state(p+off) & (STATE_COORD_VALID | STATE_LOC_VALID)) == 0)
        return 0;

    if (!problem.HasParameterBlock(&points(p)[0]))
        return 0;
    if (!problem.HasParameterBlock(&points(p+off)[0]))
        return 0;

    problem.AddResidualBlock(DistLossXY::Create(unit*cv::norm(off), dist_loss_xy_w),
                             nullptr, &points(p)[0], &points(p+off)[0]);

    return 1;
}

static int cond_surftrack_distloss_3D(int type, QuadSurface *sm, cv::Mat_<cv::Vec3d> &points, const cv::Vec2i &p, const cv::Vec2i &off,
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

static int cond_surftrack_distloss(int type, QuadSurface *sm, const cv::Vec2i &p, const cv::Vec2i &off, SurfTrackerData &data,
    ceres::Problem &problem, const cv::Mat_<uint8_t> &state, float unit, int flags = 0)
{
    resId_t id(type, sm, p, p+off);
    if (data.hasResId(id))
        return 0;

    add_surftrack_distloss(sm, p, off, data, problem, state, unit, flags, &data.resId(id));

    return 1;
}

static int add_surftrack_straightloss(QuadSurface *sm, const cv::Vec2i &p, const cv::Vec2i &o1, const cv::Vec2i &o2, const cv::Vec2i &o3,
    SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state, int flags = 0, float w = 0.7f)
{
    if (!point_in_bounds(state, p + o1) || !point_in_bounds(state, p + o2) || !point_in_bounds(state, p + o3))
        return 0;
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
    if (!point_in_bounds(state, p + o1) || !point_in_bounds(state, p + o2) || !point_in_bounds(state, p + o3))
        return 0;
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

static int cond_surftrack_straightloss_3D(int type, QuadSurface *sm, const cv::Vec2i &p, const cv::Vec2i &o1, const cv::Vec2i &o2, const cv::Vec2i &o3,
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

static int add_surftrack_surfloss(QuadSurface *sm, const cv::Vec2i& p, SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state,
    cv::Mat_<cv::Vec3d> &points, float step, ceres::ResidualBlockId *res = nullptr, float w = 0.1)
{
    if ((state(p) & STATE_LOC_VALID) == 0 || !data.valid_int(sm, p))
        return 0;

    ceres::ResidualBlockId tmp = problem.AddResidualBlock(SurfaceLossD::Create(sm->rawPoints(), w), nullptr,
                                                          &points(p)[0], &data.loc(sm, p)[0]);

    if (res)
        *res = tmp;

    return 1;
}

//gen straigt loss given point and 3 offsets
static int cond_surftrack_surfloss(int type, QuadSurface *sm, const cv::Vec2i& p, SurfTrackerData &data, ceres::Problem &problem,
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
static int surftrack_add_local(QuadSurface *sm, const cv::Vec2i& p, SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points, float step, float src_step, int flags = 0, int *straigh_count_ptr = nullptr)
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

        // XY distance for same-row neighbors (complements z_loc_loss)
        if (dist_loss_xy_w > 0.0f) {
            count += add_surftrack_distloss_XY(points, p, {0,1}, problem, state, step*src_step, flags);
            count += add_surftrack_distloss_XY(points, p, {0,-1}, problem, state, step*src_step, flags);
        }

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

    if (normal_grid_volume && (normal_loss_w > 0.0f || snap_loss_w > 0.0f)) {
        count += add_normal_grid_loss(problem, p, state, points, flags);
        count += add_normal_grid_loss(problem, p + cv::Vec2i(-1, -1), state, points, flags);
        count += add_normal_grid_loss(problem, p + cv::Vec2i(0, -1), state, points, flags);
        count += add_normal_grid_loss(problem, p + cv::Vec2i(-1, 0), state, points, flags);
    }

    if (heading_loss_w > 0.0f && heading_window > 0) {
        count += add_heading_loss_axis(points, state, p, {0,-1}, {0,1}, problem, flags);
        count += add_heading_loss_axis(points, state, p, {-1,0}, {1,0}, problem, flags);
    }

    if (flags & LOSS_ZLOC)
        problem.AddResidualBlock(ZLocationLoss<cv::Vec3f>::Create(
            sm->rawPoints(),
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
static int surftrack_add_global(QuadSurface *sm, const cv::Vec2i& p, SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state,
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

        // XY distance for same-row neighbors (complements z_loc_loss)
        if (dist_loss_xy_w > 0.0f) {
            count += add_surftrack_distloss_XY(points, p, {0,1}, problem, state, step, flags);
            count += add_surftrack_distloss_XY(points, p, {0,-1}, problem, state, step, flags);
        }

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

    if (normal_grid_volume && (normal_loss_w > 0.0f || snap_loss_w > 0.0f))
        count += add_normal_grid_loss(problem, p, state, points, flags);

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

static double local_cost_destructive(QuadSurface *sm, const cv::Vec2i& p, SurfTrackerData &data, cv::Mat_<uint8_t> &state,
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


static double local_cost(QuadSurface *sm, const cv::Vec2i& p, SurfTrackerData &data, const cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points,
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

static double local_solve(QuadSurface *sm, const cv::Vec2i &p, SurfTrackerData &data, const cv::Mat_<uint8_t> &state,
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
    bool inpaint = false,
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

                    for(int sy=0;sy<=step;sy++)
                        for(int sx=0;sx<=step;sx++) {
                            float fx = sx/step;
                            float fy = sy/step;
                            cv::Vec2f l0 = (1-fx)*l00 + fx*l01;
                            cv::Vec2f l1 = (1-fx)*l10 + fx*l11;
                            cv::Vec2f l = (1-fy)*l0 + fy*l1;
                            if (loc_valid(sm->rawPoints(), l)) {
                                points_hr(j*step+sy,i*step+sx) += cv::Vec3f(SurfTrackerData::lookup_int_loc(sm,l));
                                weights_hr(j*step+sy,i*step+sx) += 1.0f;
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

    std::cout << "hr_gen: done" << std::endl;
    return points_hr;
}

static void thin_plate_inpaint(cv::Mat_<cv::Vec3d> &points,
                               const cv::Mat_<uint8_t> &state,
                               const cv::Mat_<uint8_t> &inpaint_mask,
                               const cv::Rect &used_area,
                               int warmup_iters,
                               int thin_plate_iters)
{
    if (used_area.width <= 0 || used_area.height <= 0)
        return;

    constexpr uint8_t STATE_VALID = STATE_LOC_VALID | STATE_COORD_VALID;

    const int y0 = used_area.y;
    const int y1 = used_area.br().y;
    const int x0 = used_area.x;
    const int x1 = used_area.br().x;
    const int max_y = state.rows - 1;
    const int max_x = state.cols - 1;

    auto clamp_i = [](int v, int lo, int hi) {
        return std::max(lo, std::min(v, hi));
    };

    auto sample = [&](int yy, int xx, const cv::Vec3d &fallback) -> cv::Vec3d {
        yy = clamp_i(yy, 0, max_y);
        xx = clamp_i(xx, 0, max_x);
        if ((state(yy, xx) & STATE_VALID) == 0)
            return fallback;
        return points(yy, xx);
    };

    // Warmup: diffuse boundary values into holes to seed the biharmonic solve.
    for (int iter = 0; iter < warmup_iters; ++iter) {
        for (int y = y0; y < y1; ++y)
            for (int x = x0; x < x1; ++x)
                if (inpaint_mask(y, x)) {
                    const cv::Vec3d center = points(y, x);
                    const cv::Vec3d sum =
                        sample(y - 1, x, center) +
                        sample(y + 1, x, center) +
                        sample(y, x - 1, center) +
                        sample(y, x + 1, center);
                    points(y, x) = sum * 0.25;
                }
    }

    // Thin-plate (bi-Laplacian) smoothing using a 13-point stencil.
    for (int iter = 0; iter < thin_plate_iters; ++iter) {
        for (int y = y0; y < y1; ++y)
            for (int x = x0; x < x1; ++x)
                if (inpaint_mask(y, x)) {
                    const cv::Vec3d center = points(y, x);
                    const cv::Vec3d up = sample(y - 1, x, center);
                    const cv::Vec3d down = sample(y + 1, x, center);
                    const cv::Vec3d left = sample(y, x - 1, center);
                    const cv::Vec3d right = sample(y, x + 1, center);

                    const cv::Vec3d diag =
                        sample(y - 1, x - 1, center) +
                        sample(y - 1, x + 1, center) +
                        sample(y + 1, x - 1, center) +
                        sample(y + 1, x + 1, center);

                    const cv::Vec3d far =
                        sample(y - 2, x, center) +
                        sample(y + 2, x, center) +
                        sample(y, x - 2, center) +
                        sample(y, x + 2, center);

                    points(y, x) = (8.0 * (up + down + left + right) - 2.0 * diag - far) / 20.0;
                }
    }
}




//try flattening the current surface mapping assuming direct 3d distances
//this is basically just a reparametrization
static void optimize_surface_mapping(SurfTrackerData &data, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points, cv::Mat_<uint16_t> &generations, cv::Rect used_area,
    cv::Rect static_bounds, float step, float src_step, const cv::Vec2i &seed, int closing_r,
    const std::unordered_map<QuadSurface*, SurfPtrSet>& overlapping_map,
    SurfacePatchIndex* surface_patch_index = nullptr,
    bool save_inp_hr = true,
    const std::filesystem::path& tgt_dir = std::filesystem::path(),
    bool hr_gen_parallel = true,
    bool remap_parallel = false)
{
    std::cout << "optimizer: optimizing surface " << state.size() << " " << used_area <<  " " << static_bounds << std::endl;

    const int step_int = surftrack_round_step(step);

    cv::Mat_<cv::Vec3d> points_new = points.clone();
    cv::Mat_<cv::Vec3f> points_f;
    points.convertTo(points_f, CV_32FC3);
    QuadSurface* sm = new QuadSurface(points_f, {1,1});

    std::shared_mutex mutex;

    double pointTo_total_ms = 0.0;

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

    //     // Enable mixed precision for SPARSE_SCHUR
    //     if (options.linear_solver_type == ceres::SPARSE_SCHUR) {
    //         options.use_mixed_precision_solves = true;
    //     }
    // } else {
    //     std::cerr << "Warning: CUDA_SPARSE requested but Ceres was not built with CUDA sparse support. Falling back to default solver." << std::endl;
     }
#endif
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 100;
    options.num_threads = omp_get_max_threads();
    options.use_nonmonotonic_steps = false;

    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++)
            if (state(j,i) & STATE_LOC_VALID) {
                data_new.surfs({j,i}).insert(sm);
                data_new.loc(sm, {j,i}) = {static_cast<double>(j),static_cast<double>(i)};
            }

    cv::Mat_<uint8_t> new_state = state.clone();

    //generate closed version of state
    cv::Mat m = cv::getStructuringElement(cv::MORPH_RECT, {3,3});

    uint8_t STATE_VALID = STATE_LOC_VALID | STATE_COORD_VALID;

    int res_count = 0;
    cv::Mat_<uint8_t> inpaint_mask(state.size(), static_cast<uint8_t>(0));
    int inpainted_count = 0;
    // Build inpaint mask by closing valid regions and marking newly included cells.
    for(int r=0;r<closing_r+2;r++) {
        cv::Mat_<uint8_t> masked;
        bitwise_and(state, STATE_VALID, masked);
        cv::dilate(masked, masked, m, {-1,-1}, r);
        cv::erode(masked, masked, m, {-1,-1}, std::min(r,closing_r));
        // cv::imwrite("masked.tif", masked);

        for(int j=used_area.y;j<used_area.br().y;j++)
            for(int i=used_area.x;i<used_area.br().x;i++)
                if ((masked(j,i) & STATE_VALID) && ((new_state(j,i) & STATE_VALID) == 0)) {
                    new_state(j, i) = STATE_COORD_VALID;
                    if (points_new(j, i)[0] == -1)
                        points_new(j, i) = {0,0,0};
                    inpaint_mask(j, i) = 255;
                    inpainted_count++;
                }
    }

    if (inpainted_count > 0) {
        const int warmup_iters = std::max(10, closing_r);
        const int thin_plate_iters = std::max(50, closing_r * 10);
        std::cout << "optimizer: thin-plate inpaint " << inpainted_count
                  << " cells (warmup " << warmup_iters
                  << ", iters " << thin_plate_iters << ")" << std::endl;
        thin_plate_inpaint(points_new, new_state, inpaint_mask, used_area, warmup_iters, thin_plate_iters);

        for(int j=used_area.y;j<used_area.br().y;j++)
            for(int i=used_area.x;i<used_area.br().x;i++)
                if (inpaint_mask(j, i))
                    res_count += surftrack_add_global(sm, {j,i}, data_new, problem_inpaint, new_state, points_new, step*src_step, LOSS_3D_INDIRECT | OPTIMIZE_ALL);
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
    cv::Mat_<cv::Vec3f> points_inpainted_f;
    points_inpainted.convertTo(points_inpainted_f, CV_32FC3);
    QuadSurface* sm_inp = new QuadSurface(points_inpainted_f, {1,1});

    SurfTrackerData data_inp;
    data_inp._data = data_new._data;

    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++)
            if (new_state(j,i) & STATE_VALID) {
                data_inp.surfs({j,i}).insert(sm_inp);
                data_inp.loc(sm_inp, {j,i}) = {static_cast<double>(j),static_cast<double>(i)};
            }

    ceres::Problem problem;

    std::cout << "optimizer: using " << used_area.tl() << used_area.br() << std::endl;

    int fix_points = 0;
    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++) {
            if (!(new_state(j,i) & STATE_LOC_VALID)) continue;  // skip cells without valid locs
            res_count += surftrack_add_global(sm_inp, {j,i}, data_inp, problem, new_state, points_new, step*src_step, LOSS_3D_INDIRECT | SURF_LOSS | OPTIMIZE_ALL);
            fix_points++;
            if (problem.HasParameterBlock(&data_inp.loc(sm_inp, {j,i})[0]))
                problem.AddResidualBlock(LinChkDistLoss::Create(data_inp.loc(sm_inp, {j,i}), 1.0), nullptr, &data_inp.loc(sm_inp, {j,i})[0]);
        }

    std::cout << "optimizer: num fix points " << fix_points << std::endl;

    data_inp.seed_loc = seed;
    data_inp.seed_coord = points_new(seed);

    int fix_points_z = 0;
    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++) {
            if (!(new_state(j,i) & STATE_LOC_VALID)) continue;  // skip cells without valid locs
            fix_points_z++;
            if (problem.HasParameterBlock(&data_inp.loc(sm_inp, {j,i})[0]))
                problem.AddResidualBlock(ZLocationLoss<cv::Vec3d>::Create(points_new, data_inp.seed_coord[2] - (j-data.seed_loc[0])*step*src_step, z_loc_loss_w), new ceres::HuberLoss(1.0), &data_inp.loc(sm_inp, {j,i})[0]);
        }

    std::cout << "optimizer: optimizing " << res_count << " residuals, seed " << seed << std::endl;

    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++)
            if (static_bounds.contains(cv::Point(i,j))) {
                if (problem.HasParameterBlock(&data_inp.loc(sm_inp, {j,i})[0]))
                    problem.SetParameterBlockConstant(&data_inp.loc(sm_inp, {j,i})[0]);
                if (problem.HasParameterBlock(&points_new(j, i)[0]))
                    problem.SetParameterBlockConstant(&points_new(j, i)[0]);
            }

    options.max_num_iterations = 1000;
    options.use_nonmonotonic_steps = false;
    options.use_inner_iterations = false;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
    std::cout << "optimizer: rms " << sqrt(summary.final_cost/summary.num_residual_blocks) << " count " << summary.num_residual_blocks << std::endl;

    if (save_inp_hr) {
        cv::Mat_<cv::Vec3f> points_hr_inp =
            surftrack_genpoints_hr(data, new_state, points_inpainted, used_area, step, src_step,
                                   /*inpaint=*/true,
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
                               /*inpaint=*/false,
                               /*parallel=*/hr_gen_parallel);
    SurfTrackerData data_out;
    cv::Mat_<cv::Vec3d> points_out(points.size(), {-1,-1,-1});
    cv::Mat_<uint8_t> state_out(state.size(), 0);

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
                mutex.unlock();
            }
            else if (new_state(j,i) & STATE_VALID) {
                cv::Vec2d l = data_inp.loc(sm_inp ,{j,i});
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

                    points_out(j, i) = interp_lin_2d(points_hr, l);
                    state_out(j, i) = STATE_LOC_VALID | STATE_COORD_VALID;

                    SurfPtrSet surfs;
                    const auto& s00 = data.surfsC({y, x});
                    const auto& s01 = data.surfsC({y, x + 1});
                    const auto& s10 = data.surfsC({y + 1, x});
                    const auto& s11 = data.surfsC({y + 1, x + 1});
                    surfs.insert(s00.begin(), s00.end());
                    surfs.insert(s01.begin(), s01.end());
                    surfs.insert(s10.begin(), s10.end());
                    surfs.insert(s11.begin(), s11.end());

                    for (auto& s : surfs) {
                        const float thr = remap_attach_surface_th;
                        // use the same threshold inside pointTo
                        auto _t0 = std::chrono::steady_clock::now();
                        cv::Vec3f ptr;
                        float res = pointTo_seeded_neighbor(s, data, state, {j, i}, points_out(j, i),
                                                            thr, point_to_max_iters, surface_patch_index, &ptr);
                        double _elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _t0).count();
                        #pragma omp atomic
                        pointTo_total_ms += _elapsed;
                        if (res <= thr) {
                            cv::Vec3f loc = s->loc_raw(ptr);
                            mutex.lock();
                            data_out.surfs({j,i}).insert(s);
                            data_out.loc(s, {j,i}) = {loc[1], loc[0]};
                            mutex.unlock();
                        }
                    }

                    // No fallback attachment; if nothing attaches, orphan cleanup below will drop the point.
                }
            }
    std::cout << "remap: done" << std::endl;

    //now filter by consistency
    for(int j=used_area.y;j<used_area.br().y-1;j++)
        for(int i=used_area.x;i<used_area.br().x-1;i++)
            if (!static_bounds.contains(cv::Point(i,j)) && state_out(j,i) & STATE_VALID) {
                SurfPtrSet surf_src = data_out.surfs({j,i});
                for (auto s : surf_src) {
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
                    SurfPtrSet surf_cands = data_out.surfs({j,i});
                    for(auto s : data_out.surfs({j,i})) {
                        auto it = overlapping_map.find(s);
                        if (it != overlapping_map.end())
                            surf_cands.insert(it->second.begin(), it->second.end());
                    }
                    mutex.unlock_shared();

                    for(auto test_surf : surf_cands) {
                        mutex.lock_shared();
                        if (data_out.has(test_surf, {j,i})) {
                            mutex.unlock_shared();
                            continue;
                        }
                        mutex.unlock_shared();

                        cv::Vec3f ptr;
                        bool has_seed = false;
                        if (seed_pointto_from_neighbors) {
                            std::shared_lock<std::shared_mutex> lock(mutex);
                            has_seed = seed_ptr_from_neighbors(test_surf, data_out, state_out, {j, i},
                                                               points_out(j, i), &ptr);
                        }
                        if (!has_seed)
                            ptr = test_surf->pointer();
                        auto _t0 = std::chrono::steady_clock::now();
                        float _res = test_surf->pointTo(ptr, points_out(j, i), remap_attach_surface_th, point_to_max_iters, surface_patch_index);
                        double _elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _t0).count();
                        #pragma omp atomic
                        pointTo_total_ms += _elapsed;
                        if (_res > remap_attach_surface_th)
                            continue;

                        cv::Vec3f loc_3d = test_surf->loc_raw(ptr);
                        int count = 0;
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

    //reset unsupported points
#pragma omp parallel for
    for(int j=used_area.y;j<used_area.br().y-1;j++)
        for(int i=used_area.x;i<used_area.br().x-1;i++)
            if (!static_bounds.contains(cv::Point(i,j))) {
                if (state_out(j,i) & STATE_LOC_VALID) {
                    if (data_out.surfs({j,i}).empty()) {
                        state_out(j,i) = 0;
                        points_out(j, i) = {-1,-1,-1};
                    }
                }
                else {
                    state_out(j,i) = 0;
                    points_out(j, i) = {-1,-1,-1};
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

    // Cleanup temporary QuadSurface objects to avoid memory leak
    delete sm;
    delete sm_inp;

    dbg_counter++;
}

QuadSurface *grow_surf_from_surfs(QuadSurface *seed, const std::vector<QuadSurface*> &surfs_v, const nlohmann::json &params, float voxelsize)
{
    // Timing accumulator for pointTo calls (thread-safe via omp atomic)
    double pointTo_total_ms = 0.0;

    bool flip_x = params.value("flip_x", 0);
    int global_steps_per_window = params.value("global_steps_per_window", 0);


    std::cout << "global_steps_per_window: " << global_steps_per_window << std::endl;
    std::cout << "flip_x: " << flip_x << std::endl;
    std::filesystem::path tgt_dir = params["tgt_dir"];

    std::unordered_map<std::string,QuadSurface*> surfs;
    float src_step = params.value("src_step", 20);
    float step = params.value("step", 10);
    const int step_int = surftrack_round_step(step);
    int max_width = params.value("max_width", 80000);
    std::string normal_grid_path;

    local_cost_inl_th = params.value("local_cost_inl_th", 0.2f);
    assoc_surface_th = params.value("same_surface_th", 2.0f);
    duplicate_surface_th = params.value("duplicate_surface_th", assoc_surface_th);
    remap_attach_surface_th = params.value("remap_attach_surface_th", assoc_surface_th);
    point_to_max_iters = params.value("point_to_max_iters", 10);
    int point_to_seed_max_iters = params.value("point_to_seed_max_iters", 1000);
    bool use_surface_patch_index = params.value("use_surface_patch_index", false);
    int surface_patch_stride = params.value("surface_patch_stride", 1);
    float surface_patch_bbox_pad = params.value("surface_patch_bbox_pad", 0.0f);
    float surface_index_min_radius = params.value("surface_index_search_radius", 100.0f);
    if (surface_index_min_radius < 0.0f)
        surface_index_min_radius = 0.0f;
    int max_local_surfs = params.value("max_local_surfs", 0);
    if (max_local_surfs < 0)
        max_local_surfs = 0;
    seed_pointto_from_neighbors = params.value("seed_pointto_from_neighbors", false);
    std::string pointto_interp = params.value("pointto_interp", std::string("bilinear"));
    bool pointto_catmull = false;
    if (pointto_interp == "catmull-rom" || pointto_interp == "catmull_rom" || pointto_interp == "catmull") {
        pointto_catmull = true;
    } else if (pointto_interp != "bilinear" && pointto_interp != "linear") {
        std::cerr << "WARNING: pointto_interp must be 'bilinear' or 'catmull-rom'; defaulting to bilinear" << std::endl;
        pointto_interp = "bilinear";
    }
    QuadSurface::setPointToCatmullRom(pointto_catmull);
    straight_weight = params.value("straight_weight", 0.7f);            // Weight for 2D straight line constraints
    straight_weight_3D = params.value("straight_weight_3D", 4.0f);      // Weight for 3D straight line constraints
    sliding_w_scale = params.value("sliding_w_scale", 1.0f);            // Scale factor for sliding window
    z_loc_loss_w = params.value("z_loc_loss_w", 0.1f);                  // Weight for Z location loss constraints
    dist_loss_2d_w = params.value("dist_loss_2d_w", 1.0f);              // Weight for 2D distance constraints
    dist_loss_3d_w = params.value("dist_loss_3d_w", 2.0f);              // Weight for 3D distance constraints
    dist_loss_xy_w = params.value("dist_loss_xy_w", 1.0f);              // Weight for XY distance constraints (same-row)
    straight_min_count = params.value("straight_min_count", 1.0f);      // Minimum number of straight constraints
    inlier_base_threshold = params.value("inlier_base_threshold", 20);  // Starting threshold for inliers
    heading_window = params.value("heading_window", 9);
    if (heading_window < 1)
        heading_window = 1;
    heading_min_segments = params.value("heading_min_segments", 3);
    if (heading_min_segments < 1)
        heading_min_segments = 1;
    if (heading_min_segments > heading_window)
        heading_min_segments = heading_window;
    heading_loss_w = params.value("heading_loss_w", 0.5f);
    normal_loss_w = params.value("normal_weight", 1.0f);
    snap_loss_w = params.value("snap_weight", 0.1f);
    normal_grid_z_min = params.value("normal_grid_z_min", params.value("z_min", -1));
    normal_grid_z_max = params.value("normal_grid_z_max",
                                     params.value("z_max", std::numeric_limits<int>::max()));
    if (normal_grid_z_min > normal_grid_z_max)
        std::swap(normal_grid_z_min, normal_grid_z_max);
    normal_grid_volume.reset();
    if (params.contains("normal_grid_path")) {
        try {
            normal_grid_path = params["normal_grid_path"].get<std::string>();
            normal_grid_volume = std::make_unique<vc::core::util::NormalGridVolume>(normal_grid_path);
        } catch (const std::exception& ex) {
            std::cerr << "WARNING: failed to load normal_grid_path: " << ex.what() << std::endl;
            normal_grid_volume.reset();
        }
    }
    uint64_t deterministic_seed = uint64_t(params.value("deterministic_seed", 5489));
    double deterministic_jitter_px = params.value("deterministic_jitter_px", 0.15);
    std::string candidate_ordering = params.value("candidate_ordering", "row_col");
    bool candidate_min_dist_set = params.contains("candidate_min_dist");
    int candidate_min_dist = params.value("candidate_min_dist", 0);
    if (candidate_min_dist < 0)
        candidate_min_dist = 0;
    int hole_requeue_interval = params.value("hole_requeue_interval", 100);
    int hole_seal_radius = params.value("hole_seal_radius", 2);
    int hole_max_retries = params.value("hole_max_retries", 3);
    int hole_edge_prune_min_neighbors = params.value("hole_edge_prune_min_neighbors", 0);
    int force_retry_min_neighbors = params.value("force_retry_min_neighbors", 3);
    int force_retry_max = params.value("force_retry_max", 3);
    int disconnect_prune_interval = params.value("disconnect_prune_interval", 0);
    int disconnect_window_width = params.value("disconnect_window_width", 0);
    int disconnect_window_stride = params.value("disconnect_window_stride", 0);
    int disconnect_erode_radius = params.value("disconnect_erode_radius", 0);
    if (hole_requeue_interval < 1)
        hole_requeue_interval = 0;
    if (hole_seal_radius < 0)
        hole_seal_radius = 0;
    if (hole_max_retries < 1)
        hole_max_retries = 0;
    if (hole_max_retries > 255)
        hole_max_retries = 255;
    if (hole_edge_prune_min_neighbors < 0)
        hole_edge_prune_min_neighbors = 0;
    if (force_retry_min_neighbors < 1)
        force_retry_min_neighbors = 0;
    if (force_retry_max < 1)
        force_retry_max = 0;
    if (force_retry_max > 255)
        force_retry_max = 255;
    if (disconnect_prune_interval < 1)
        disconnect_prune_interval = 0;
    if (disconnect_window_width < 0)
        disconnect_window_width = 0;
    if (disconnect_window_stride < 0)
        disconnect_window_stride = 0;
    if (disconnect_erode_radius < 0)
        disconnect_erode_radius = 0;
    bool use_spread_out_ordering = false;
    if (candidate_ordering == "legacy") {
        use_spread_out_ordering = true;
        if (!candidate_min_dist_set)
            candidate_min_dist = 9;
    } else if (candidate_ordering == "spread_out" || candidate_ordering == "spread-out") {
        use_spread_out_ordering = true;
        if (!candidate_min_dist_set)
            candidate_min_dist = 9;
        std::cout << "warning: candidate_ordering '" << candidate_ordering
                  << "' is deprecated; use 'legacy' to match SurfaceHelpers" << std::endl;
        candidate_ordering = "legacy";
    } else if (candidate_ordering != "row_col") {
        std::cout << "warning: unknown candidate_ordering '" << candidate_ordering
                  << "', using row_col" << std::endl;
        candidate_ordering = "row_col";
    }

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
    if (params.contains("z_range") && !params.contains("normal_grid_z_min") &&
        !params.contains("normal_grid_z_max") && !params.contains("z_min") &&
        !params.contains("z_max")) {
        normal_grid_z_min = static_cast<int>(std::lround(z_min));
        normal_grid_z_max = static_cast<int>(std::lround(z_max));
        if (normal_grid_z_min > normal_grid_z_max)
            std::swap(normal_grid_z_min, normal_grid_z_max);
    }

    std::cout << "  local_cost_inl_th: " << local_cost_inl_th << std::endl;
    std::cout << "  assoc_surface_th: " << assoc_surface_th << std::endl;
    std::cout << "  duplicate_surface_th: " << duplicate_surface_th << std::endl;
    std::cout << "  remap_attach_surface_th: " << remap_attach_surface_th << std::endl;
    std::cout << "  point_to_max_iters: " << point_to_max_iters << std::endl;
    std::cout << "  point_to_seed_max_iters: " << point_to_seed_max_iters << std::endl;
    std::cout << "  use_surface_patch_index: " << use_surface_patch_index << std::endl;
    std::cout << "  surface_patch_stride: " << surface_patch_stride << std::endl;
    std::cout << "  surface_patch_bbox_pad: " << surface_patch_bbox_pad << std::endl;
    std::cout << "  surface_index_search_radius: " << surface_index_min_radius << std::endl;
    std::cout << "  straight_weight: " << straight_weight << std::endl;
    std::cout << "  straight_weight_3D: " << straight_weight_3D << std::endl;
    std::cout << "  straight_min_count: " << straight_min_count << std::endl;
    std::cout << "  inlier_base_threshold: " << inlier_base_threshold << std::endl;
    std::cout << "  sliding_w_scale: " << sliding_w_scale << std::endl;
    std::cout << "  z_loc_loss_w: " << z_loc_loss_w << std::endl;
    std::cout << "  dist_loss_2d_w: " << dist_loss_2d_w << std::endl;
    std::cout << "  dist_loss_3d_w: " << dist_loss_3d_w << std::endl;
    std::cout << "  dist_loss_xy_w: " << dist_loss_xy_w << std::endl;
    std::cout << "  normal_weight: " << normal_loss_w << std::endl;
    std::cout << "  snap_weight: " << snap_loss_w << std::endl;
    std::cout << "  heading_window: " << heading_window << std::endl;
    std::cout << "  heading_min_segments: " << heading_min_segments << std::endl;
    std::cout << "  heading_loss_w: " << heading_loss_w << std::endl;
    std::cout << "  deterministic_seed: " << deterministic_seed << std::endl;
    std::cout << "  deterministic_jitter_px: " << deterministic_jitter_px << std::endl;
    std::cout << "  candidate_ordering: " << candidate_ordering << std::endl;
    std::cout << "  candidate_min_dist: " << candidate_min_dist << std::endl;
    std::cout << "  hole_requeue_interval: " << hole_requeue_interval << std::endl;
    std::cout << "  hole_seal_radius: " << hole_seal_radius << std::endl;
    std::cout << "  hole_max_retries: " << hole_max_retries << std::endl;
    std::cout << "  hole_edge_prune_min_neighbors: " << hole_edge_prune_min_neighbors << std::endl;
    std::cout << "  force_retry_min_neighbors: " << force_retry_min_neighbors << std::endl;
    std::cout << "  force_retry_max: " << force_retry_max << std::endl;
    std::cout << "  max_local_surfs: " << max_local_surfs << std::endl;
    std::cout << "  seed_pointto_from_neighbors: " << (seed_pointto_from_neighbors ? 1 : 0) << std::endl;
    std::cout << "  pointto_interp: " << (pointto_catmull ? "catmull-rom" : "bilinear") << std::endl;
    if (enforce_z_range)
        std::cout << "  z_range: [" << z_min << ", " << z_max << "]" << std::endl;
    if (normal_grid_volume) {
        std::cout << "  normal_grid_path: " << normal_grid_path << std::endl;
        if (normal_grid_z_min != -1 || normal_grid_z_max != std::numeric_limits<int>::max())
            std::cout << "  normal_grid_z_range: [" << normal_grid_z_min << ", "
                      << normal_grid_z_max << "]" << std::endl;
    }

    std::cout << "total surface count: " << surfs_v.size() << std::endl;

    std::unordered_set<QuadSurface*> approved_sm;

    std::set<std::string> used_approved_names;
    std::string log_filename = "/tmp/vc_grow_seg_from_segments_" + get_surface_time_str() + "_used_approved_segments.txt";
    std::ofstream approved_log(log_filename);

    for(auto &sm : surfs_v) {
        if (sm->meta->contains("tags") && sm->meta->at("tags").contains("approved"))
            approved_sm.insert(sm);
        if (!sm->meta->contains("tags") || !sm->meta->at("tags").contains("defective")) {
            surfs[sm->id] = sm;
        }
    }

    for(auto sm : approved_sm)
        std::cout << "approved: " << sm->id << '\n';

    // Build overlapping map: for each surface, collect pointers to its overlapping surfaces
    std::unordered_map<QuadSurface*, SurfPtrSet> overlapping_map;
    for(auto &sm : surfs_v)
        for(const auto& name : sm->overlappingIds())
            if (surfs.contains(name))
                overlapping_map[sm].insert(surfs[name]);

    SurfacePatchIndex patch_index;
    SurfacePatchIndex* patch_index_ptr = nullptr;
    if (use_surface_patch_index) {
        std::vector<SurfacePatchIndex::SurfacePtr> patch_surfaces;
        patch_surfaces.reserve(surfs.size());
        for (const auto& it : surfs)
            patch_surfaces.emplace_back(SurfacePatchIndex::SurfacePtr(it.second, [](QuadSurface*) {}));
        patch_index.setSamplingStride(surface_patch_stride);
        patch_index.setMinSearchRadius(surface_index_min_radius);
        patch_index.rebuild(patch_surfaces, surface_patch_bbox_pad);
        patch_index_ptr = &patch_index;
        std::cout << "SurfacePatchIndex built for " << patch_surfaces.size() << " surfaces" << std::endl;
    }

    std::cout << "total surface count (after defective filter): " << surfs.size() << std::endl;
    std::cout << "seed " << seed << " name " << seed->id << " seed overlapping: "
              << overlapping_map[seed].size() << "/" << seed->overlappingIds().size() << std::endl;

    cv::Mat_<cv::Vec3f> seed_points = seed->rawPoints();

    int stop_gen = 100000;
    int closing_r = 20; //FIXME dont forget to reset!

    // Get sliding window scale from params (set earlier from JSON)

    //1k ~ 1cm, scaled by sliding_w_scale parameter
    int sliding_w = static_cast<int>(1000/src_step/step*2 * sliding_w_scale);
    if (disconnect_prune_interval > 0) {
        if (disconnect_window_width < 1)
            disconnect_window_width = std::max(1, sliding_w);
        if (disconnect_window_stride < 1)
            disconnect_window_stride = disconnect_window_width;
        std::cout << "  disconnect_prune_interval: " << disconnect_prune_interval << std::endl;
        std::cout << "  disconnect_window_width: " << disconnect_window_width << std::endl;
        std::cout << "  disconnect_window_stride: " << disconnect_window_stride << std::endl;
        if (disconnect_erode_radius > 0)
            std::cout << "  disconnect_erode_radius: " << disconnect_erode_radius << std::endl;
    }
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
    cv::Mat_<uint8_t> hole_retry_count(size, static_cast<uint8_t>(0));
    cv::Mat_<uint8_t> force_retry_count(size, static_cast<uint8_t>(0));
    cv::Mat_<uint8_t> force_retry_mark(size, static_cast<uint8_t>(0));
    std::vector<cv::Vec2i> force_retry;

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
        cv::Vec3f coord = points(p);
        std::cout << "testing " << p << " from cands: " << overlapping_map[seed].size() << coord << std::endl;
        for(auto s : overlapping_map[seed]) {
            auto _t0 = std::chrono::steady_clock::now();
            cv::Vec3f ptr;
            float _res = pointTo_seeded_neighbor(s, data, state, p, coord,
                                                 assoc_surface_th, point_to_seed_max_iters, patch_index_ptr, &ptr);
            double _elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _t0).count();
            pointTo_total_ms += _elapsed;
            if (_res <= assoc_surface_th) {
                cv::Vec3f loc = s->loc_raw(ptr);
                data.surfs(p).insert(s);
                data.loc(s, p) = {loc[1], loc[0]};
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
    int global_opt_count = 0;

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

        if (!force_retry.empty()) {
            for (const auto& p : force_retry) {
                force_retry_mark(p) = 0;
                if (state(p) & STATE_LOC_VALID)
                    continue;
                if (state(p) & STATE_PROCESSING)
                    continue;
                if (!save_bounds_inv.contains(cv::Point(p)))
                    continue;
                state(p) |= STATE_PROCESSING;
                cands.insert(p);
            }
            force_retry.clear();
        }

        if (generation % 100 == 0) {
            std::cout << "go with cands " << cands.size() << " inl_th " << curr_best_inl_th << std::endl;
        }

        // Deterministic, sorted vector of candidates
        std::vector<cv::Vec2i> cands_vec(cands.begin(), cands.end());

        std::shared_mutex mutex;
        int best_inliers_gen = 0;
        const int r = 1;

        auto valid_neighbor_count = [&](const cv::Vec2i& p) {
            int count = 0;
            for (const auto& n : neighs) {
                cv::Vec2i pn = p + n;
                if (point_in_bounds(state, pn) && (state(pn) & STATE_LOC_VALID))
                    count++;
            }
            return count;
        };

        auto queue_force_retry = [&](const cv::Vec2i& p, int neighbor_count) {
            if (force_retry_max == 0 || force_retry_min_neighbors == 0)
                return;
            if (neighbor_count < force_retry_min_neighbors)
                return;
            if (!save_bounds_inv.contains(cv::Point(p)))
                return;
            std::lock_guard<std::shared_mutex> lock(mutex);
            if (force_retry_mark(p))
                return;
            if (force_retry_count(p) >= force_retry_max)
                return;
            force_retry_mark(p) = 1;
            force_retry_count(p) = static_cast<uint8_t>(force_retry_count(p) + 1);
            force_retry.push_back(p);
        };

        auto process_candidate = [&](const cv::Vec2i& p) {

            if (state(p) & STATE_LOC_VALID)
                return;

            if (points(p)[0] != -1)
                throw std::runtime_error("oops points(p)[0]");

            SurfPtrSet local_surfs;
            local_surfs.insert(seed);
            std::unordered_map<QuadSurface*, int> local_counts;
            local_counts[seed] = 1;

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
                }
            }
            if (misses) {
                std::cout << "grow: cleaned " << misses << " stale surface refs in thread-local cache" << std::endl;
            }
            mutex.unlock_shared();
            mutex.lock();
            added_points_threads[omp_get_thread_num()].resize(0);
            mutex.unlock();

            // Build local_surfs from valid neighbors
            for(int oy=std::max(p[0]-r,0);oy<=std::min(p[0]+r,h-1);oy++)
                for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,w-1);ox++)
                    if (state(oy,ox) & STATE_LOC_VALID) {
                        auto p_surfs = data_th.surfsC({oy,ox});
                        for (auto s : p_surfs) {
                            local_surfs.insert(s);
                            local_counts[s] += 1;
                        }
                    }

            std::vector<QuadSurface*> local_surfs_vec(local_surfs.begin(), local_surfs.end());
            if (max_local_surfs > 0 &&
                local_surfs_vec.size() > static_cast<size_t>(max_local_surfs)) {
                // Prefer approved + frequently-seen local surfaces when trimming.
                std::sort(local_surfs_vec.begin(), local_surfs_vec.end(),
                          [&](QuadSurface* a, QuadSurface* b) {
                              const bool a_approved = approved_sm.contains(a);
                              const bool b_approved = approved_sm.contains(b);
                              if (a_approved != b_approved)
                                  return a_approved > b_approved;
                              const auto a_it = local_counts.find(a);
                              const auto b_it = local_counts.find(b);
                              const int a_count = (a_it == local_counts.end()) ? 0 : a_it->second;
                              const int b_count = (b_it == local_counts.end()) ? 0 : b_it->second;
                              if (a_count != b_count)
                                  return a_count > b_count;
                              return a->id < b->id;
                          });
                local_surfs_vec.resize(max_local_surfs);
            }

            SurfPtrSet local_surfs_filtered(local_surfs_vec.begin(), local_surfs_vec.end());
            std::vector<QuadSurface*> test_surfs = local_surfs_vec;

            std::vector<QuadSurface*> ref_surfs = local_surfs_vec;
            std::sort(ref_surfs.begin(), ref_surfs.end(),
                      [&](QuadSurface* a, QuadSurface* b) {
                          const bool a_approved = approved_sm.contains(a);
                          const bool b_approved = approved_sm.contains(b);
                          if (a_approved != b_approved)
                              return a_approved > b_approved;
                          return a->id < b->id;
                      });

            cv::Vec3d best_coord = {-1,-1,-1};
            int best_inliers = -1;
            QuadSurface *best_surf = nullptr;
            cv::Vec2d best_loc = {-1,-1};
            bool best_ref_seed = false;
            for(auto ref_surf : ref_surfs) {
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
                uint64_t surf_key = mix64(uint64_t(std::hash<std::string>{}(ref_surf->id)));
                uint64_t salt = deterministic_seed ^ surf_key;
                double j0 = det_jitter_symm(p[0], p[1], salt) * deterministic_jitter_px;
                double j1 = det_jitter_symm(p[0], p[1], salt ^ 0x9e3779b97f4a7c15ULL) * deterministic_jitter_px;
                data_th.loc(ref_surf,p) = avg + cv::Vec2d(j0, j1);

                ceres::Problem problem;

                state(p) = STATE_LOC_VALID | STATE_COORD_VALID;

                int straight_count_init = 0;
                int count_init = surftrack_add_local(ref_surf, p, data_th, problem, state, points, step, src_step, LOSS_ZLOC, &straight_count_init);
                if (count_init == 0) {
                    state(p) = 0;
                    generations(p) = 0;
                    data_th.erase(ref_surf, p);
                    continue;
                }
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

                if (approved_sm.contains(ref_surf) && straight_count_init >= 2 && count_init >= 4) {
                    best_inliers = 1000;
                    best_coord = coord;
                    best_surf = ref_surf;
                    best_loc = ref_loc;
                    best_ref_seed = ref_seed;
                    data_th.erase(ref_surf, p);
                    break;
                }

                int inliers_sum = 0;
                int inliers_count = 0;

                for(auto test_surf : test_surfs) {
                    //FIXME this does not check geometry, only if its also on the surfaces (which might be good enough...)
                    auto _t0 = std::chrono::steady_clock::now();
                    cv::Vec3f ptr;
                    float _res = pointTo_seeded_neighbor(test_surf, data_th, state, p, coord,
                                                         assoc_surface_th, point_to_max_iters, patch_index_ptr, &ptr);
                    double _elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _t0).count();
                    #pragma omp atomic
                    pointTo_total_ms += _elapsed;
                    if (_res <= assoc_surface_th) {
                        cv::Vec3f loc = test_surf->loc_raw(ptr);
                        int count = 0;
                        int straight_count = 0;
                        state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
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

            // Guard against duplicating existing 3D coords.
            if (best_inliers >= curr_best_inl_th || best_ref_seed)
            {
                if (enforce_z_range && (best_coord[2] < z_min || best_coord[2] > z_max)) {
                    // Final guard: reject best candidate outside z-range
                    best_inliers = -1;
                    best_ref_seed = false;
                } else if (used_area.width >=4 && used_area.height >= 4) {
                    cv::Vec2f tmp_loc_;
                    cv::Rect used_th = used_area;
                    float dist = pointTo(tmp_loc_, points(used_th), best_coord, duplicate_surface_th, 1000, 1.0/(step*src_step));
                    tmp_loc_ += cv::Vec2f(used_th.x,used_th.y);
                    if (dist <= duplicate_surface_th) {
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
                state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
                points(p) = best_coord;
                hole_retry_count(p) = 0;
                force_retry_count(p) = 0;
                force_retry_mark(p) = 0;
                inliers_sum_dbg(p) = best_inliers;
                generations(p) = static_cast<uint16_t>(std::min<int>(std::numeric_limits<uint16_t>::max(), generation + 1));
                if (approved_sm.contains(best_surf)) {
                    mutex.lock();
                    if (used_approved_names.insert(best_surf->id).second)
                        approved_log << best_surf->id << '\n';
                    mutex.unlock();
                }

                ceres::Problem problem;
                surftrack_add_local(best_surf, p, data_th, problem, state, points, step, src_step, SURF_LOSS | LOSS_ZLOC);

                SurfPtrSet more_local_surfs;

                for(auto test_surf : test_surfs) {
                    for(auto s : overlapping_map[test_surf])
                        if (!local_surfs_filtered.contains(s) && s != best_surf)
                            more_local_surfs.insert(s);

                    if (test_surf == best_surf)
                        continue;

                    auto _t0 = std::chrono::steady_clock::now();
                    cv::Vec3f ptr;
                    float _res = pointTo_seeded_neighbor(test_surf, data_th, state, p, best_coord,
                                                         assoc_surface_th, point_to_max_iters, patch_index_ptr, &ptr);
                    double _elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _t0).count();
                    #pragma omp atomic
                    pointTo_total_ms += _elapsed;
                    if (_res <= assoc_surface_th) {
                        cv::Vec3f loc = test_surf->loc_raw(ptr);
                        cv::Vec3f coord = SurfTrackerData::lookup_int_loc(test_surf, {loc[1], loc[0]});
                        if (coord[0] == -1) {
                            continue;
                        }
                        int count = 0;
                        float cost = local_cost_destructive(test_surf, p, data_th, state, points, step, src_step, loc, &count);
                        if (cost < local_cost_inl_th) {
                            data_th.loc(test_surf, p) = {loc[1], loc[0]};
                            data_th.surfs(p).insert(test_surf);
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
                    auto _t0 = std::chrono::steady_clock::now();
                    cv::Vec3f ptr;
                    float res = pointTo_seeded_neighbor(test_surf, data_th, state, p, best_coord,
                                                        assoc_surface_th, point_to_max_iters, patch_index_ptr, &ptr);
                    double _elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _t0).count();
                    #pragma omp atomic
                    pointTo_total_ms += _elapsed;
                    if (res <= assoc_surface_th) {
                        cv::Vec3f loc = test_surf->loc_raw(ptr);
                        cv::Vec3f coord = SurfTrackerData::lookup_int_loc(test_surf, {loc[1], loc[0]});
                        if (coord[0] == -1) {
                            continue;
                        }
                        int count = 0;
                        float cost = local_cost_destructive(test_surf, p, data_th, state, points, step, src_step, loc, &count);
                        if (cost < local_cost_inl_th) {
                            data_th.loc(test_surf, p) = {loc[1], loc[0]};
                            data_th.surfs(p).insert(test_surf);
                        };
                    }
                }

                mutex.lock();
                succ++;

                // Rebuild global set only from surfaces that have a valid loc in thread-local
                SurfPtrSet accepted;
                for (auto &s : data_th.surfs(p))
                    if (data_th.has(s, p))
                        accepted.insert(s);

                data.surfs(p).clear();
                for (auto &s : accepted) {
                    data.surfs(p).insert(s);
                    data.loc(s, p) = data_th.loc(s, p);
                }
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
                queue_force_retry(p, valid_neighbor_count(p));
            }
            else {
                state(p) = 0;
                points(p) = {-1,-1,-1};
                generations(p) = 0;
                queue_force_retry(p, valid_neighbor_count(p));
#pragma omp critical
                best_inliers_gen = std::max(best_inliers_gen, best_inliers);
            }
        };

        auto process_cands_row_col = [&]() {
            // Column-wise processing: grow left-to-right by column (x).
            std::map<int, std::vector<cv::Vec2i>> by_col;
            for (const auto& p : cands_vec)
                by_col[p[1]].push_back(p);

            for (auto &kv : by_col) {
#pragma omp parallel for schedule(static)
                for (int idx = 0; idx < static_cast<int>(kv.second.size()); ++idx)
                    process_candidate(kv.second[idx]);
            }
        };

        auto process_cands_spread = [&]() {
            OmpThreadPointCol cands_threadcol(candidate_min_dist, cands_vec);
#pragma omp parallel
            {
                while (true) {
                    cv::Vec2i p = cands_threadcol.next();
                    if (p[0] == -1)
                        break;
                    process_candidate(p);
                }
            }
        };

        if (use_spread_out_ordering)
            process_cands_spread();
        else
            process_cands_row_col();

        if (generation == 1 && flip_x) {
            data.flip_x(x0);

            for(int i=0;i<omp_get_max_threads();i++) {
                data_ths[i] = data;
                added_points_threads[i].clear();
            }

            cv::Mat_<uint8_t> state_orig = state.clone();
            cv::Mat_<cv::Vec3d> points_orig = points.clone();
            cv::Mat_<uint16_t> generations_orig = generations.clone();
            cv::Mat_<uint8_t> hole_retry_count_orig = hole_retry_count.clone();
            state.setTo(0);
            points.setTo(cv::Vec3d(-1,-1,-1));
            generations.setTo(0);
            hole_retry_count.setTo(0);
            force_retry_count.setTo(0);
            force_retry_mark.setTo(0);
            force_retry.clear();
            cv::Rect new_used_area = used_area;
            for(int j=used_area.y;j<=used_area.br().y+1;j++)
                for(int i=used_area.x;i<=used_area.br().x+1;i++)
                    if (state_orig(j, i)) {
                        int nx = x0+x0-i;
                        int ny = j;
                        state(ny, nx) = state_orig(j, i);
                        points(ny, nx) = points_orig(j, i);
                        generations(ny, nx) = generations_orig(j, i);
                        hole_retry_count(ny, nx) = hole_retry_count_orig(j, i);
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

        if (hole_requeue_interval > 0 &&
            hole_max_retries > 0 &&
            generation > 0 &&
            (generation % hole_requeue_interval) == 0) {
            cv::Rect hole_roi(0, 0, state.cols, state.rows);
            if (hole_roi.width > 0 && hole_roi.height > 0) {
                cv::Mat valid_mask(hole_roi.height, hole_roi.width, CV_8U, cv::Scalar(0));
                for (int y = 0; y < hole_roi.height; ++y) {
                    uint8_t* row = valid_mask.ptr<uint8_t>(y);
                    for (int x = 0; x < hole_roi.width; ++x) {
                        if (state(y, x) & STATE_LOC_VALID)
                            row[x] = 255;
                    }
                }

                cv::Mat kernel = cv::getStructuringElement(
                    cv::MORPH_RECT,
                    cv::Size(hole_seal_radius * 2 + 1, hole_seal_radius * 2 + 1));
                cv::Mat closed_valid;
                // Close valid to seal narrow outside channels before flood fill.
                cv::morphologyEx(valid_mask, closed_valid, cv::MORPH_CLOSE, kernel);

                cv::Mat invalid_orig;
                cv::Mat invalid_closed;
                cv::bitwise_not(valid_mask, invalid_orig);
                cv::bitwise_not(closed_valid, invalid_closed);

                cv::Mat padded;
                cv::copyMakeBorder(invalid_closed, padded, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
                cv::floodFill(padded, cv::Point(0, 0), 128);
                cv::Mat outside = padded(cv::Rect(1, 1, hole_roi.width, hole_roi.height)) == 128;
                cv::Mat outside_inv;
                cv::bitwise_not(outside, outside_inv);

                cv::Mat holes;
                // Use original invalid mask to keep true hole geometry.
                cv::bitwise_and(invalid_orig, outside_inv, holes);

                int edge_pruned = 0;
                if (cv::countNonZero(holes) > 0) {
                    if (hole_edge_prune_min_neighbors > 0) {
                        cv::Mat holes_dilated;
                        cv::dilate(holes, holes_dilated, cv::getStructuringElement(cv::MORPH_RECT, {3,3}));
                        cv::Mat hole_adj_valid;
                        cv::bitwise_and(holes_dilated, valid_mask, hole_adj_valid);

                        std::vector<cv::Vec2i> prune_points;
                        prune_points.reserve(cv::countNonZero(hole_adj_valid));
                        for (int y = 0; y < hole_adj_valid.rows; ++y) {
                            const uint8_t* row = hole_adj_valid.ptr<uint8_t>(y);
                            for (int x = 0; x < hole_adj_valid.cols; ++x) {
                                if (!row[x])
                                    continue;
                                cv::Vec2i p = {y, x};
                                if ((state(p) & STATE_LOC_VALID) == 0)
                                    continue;
                                int neighbor_count = 0;
                                for (const auto& n : neighs) {
                                    cv::Vec2i pn = p + n;
                                    if (point_in_bounds(state, pn) && (state(pn) & STATE_LOC_VALID))
                                        neighbor_count++;
                                }
                                if (neighbor_count < hole_edge_prune_min_neighbors)
                                    prune_points.push_back(p);
                            }
                        }

                        if (!prune_points.empty()) {
                            auto clear_point = [&](const cv::Vec2i& p) {
                                if (state(p) & STATE_LOC_VALID) {
                                    auto &surfs_here = data.surfs(p);
                                    for (auto s : surfs_here)
                                        data.erase(s, p);
                                    surfs_here.clear();
                                }
                                state(p) = 0;
                                points(p) = {-1,-1,-1};
                                generations(p) = 0;
                                inliers_sum_dbg(p) = 0;
                                hole_retry_count(p) = 0;
                                force_retry_count(p) = 0;
                                force_retry_mark(p) = 0;
                            };

                            for (const auto& p : prune_points) {
                                clear_point(p);
                                valid_mask.at<uint8_t>(p[0], p[1]) = 0;
                                holes.at<uint8_t>(p[0], p[1]) = 255;
                            }
                            edge_pruned = static_cast<int>(prune_points.size());
                        }
                    }

                    cv::Mat eligible_holes(hole_roi.height, hole_roi.width, CV_8U, cv::Scalar(0));
                    int eligible = 0;
                    for (int y = 0; y < hole_roi.height; ++y) {
                        const uint8_t* hole_row = holes.ptr<uint8_t>(y);
                        uint8_t* eligible_row = eligible_holes.ptr<uint8_t>(y);
                        for (int x = 0; x < hole_roi.width; ++x) {
                            if (!hole_row[x])
                                continue;
                            if (hole_retry_count(y, x) >= hole_max_retries)
                                continue;
                            eligible_row[x] = 255;
                            hole_retry_count(y, x) = static_cast<uint8_t>(hole_retry_count(y, x) + 1);
                            eligible++;
                            if ((state(y, x) & STATE_LOC_VALID) == 0)
                                state(y, x) &= ~STATE_PROCESSING;
                        }
                    }

                    if (eligible > 0) {
                        cv::Mat holes_dilated;
                        cv::dilate(eligible_holes, holes_dilated, cv::getStructuringElement(cv::MORPH_RECT, {3,3}));
                        cv::Mat hole_adj_valid;
                        cv::bitwise_and(holes_dilated, valid_mask, hole_adj_valid);

                        int requeued = 0;
                        for (int y = 0; y < hole_roi.height; ++y) {
                            const uint8_t* adj_row = hole_adj_valid.ptr<uint8_t>(y);
                            for (int x = 0; x < hole_roi.width; ++x) {
                                if (adj_row[x]) {
                                    fringe.insert(cv::Vec2i(y, x));
                                    requeued++;
                                }
                            }
                        }
                        if (requeued > 0)
                            std::cout << "hole-seal: requeued " << requeued
                                      << " boundary points (" << eligible << " hole pixels)" << std::endl;
                    }
                }
                if (edge_pruned > 0) {
                    for(int i=0;i<omp_get_max_threads();i++) {
                        data_ths[i] = data;
                        added_points_threads[i].clear();
                    }
                    for (const auto& p : force_retry)
                        force_retry_mark(p) = 0;
                    force_retry.clear();
                    init_area_scan();
                    std::cout << "hole-edge-prune: removed " << edge_pruned << " points" << std::endl;
                }
            }
        }

        if (disconnect_prune_interval > 0 &&
            disconnect_window_width > 0 &&
            generation > 0 &&
            (generation % disconnect_prune_interval) == 0) {
            int used_x0 = std::max(0, used_area.x);
            int used_x1 = std::min(state.cols, used_area.br().x);
            int available_w = used_x1 - used_x0;
            int win_w = std::min(disconnect_window_width, available_w);
            int stride = disconnect_window_stride;
            if (stride < 1)
                stride = win_w;

            if (available_w > 0 && win_w > 0) {
                std::set<cv::Vec2i, Vec2iLess> requeue;
                int removed_total = 0;
                int windows_pruned = 0;

                std::array<cv::Vec2i, 8> neigh8 = {{
                    {1,0},{0,1},{-1,0},{0,-1},
                    {1,1},{1,-1},{-1,1},{-1,-1}
                }};
                cv::Mat disconnect_kernel;
                if (disconnect_erode_radius > 0) {
                    disconnect_kernel = cv::getStructuringElement(
                        cv::MORPH_RECT,
                        cv::Size(disconnect_erode_radius * 2 + 1, disconnect_erode_radius * 2 + 1));
                }

                auto clear_point = [&](const cv::Vec2i& p) {
                    if (state(p) & STATE_LOC_VALID) {
                        auto &surfs_here = data.surfs(p);
                        for (auto s : surfs_here)
                            data.erase(s, p);
                        surfs_here.clear();
                    }
                    state(p) = 0;
                    points(p) = {-1,-1,-1};
                    generations(p) = 0;
                    inliers_sum_dbg(p) = 0;
                    hole_retry_count(p) = 0;
                    force_retry_count(p) = 0;
                    force_retry_mark(p) = 0;
                };

                auto process_window = [&](int x_start) {
                    cv::Mat mask(state.rows, win_w, CV_8U, cv::Scalar(0));
                    for (int y = 0; y < state.rows; ++y) {
                        uint8_t* row = mask.ptr<uint8_t>(y);
                        for (int x = 0; x < win_w; ++x) {
                            if (state(y, x_start + x) & STATE_LOC_VALID)
                                row[x] = 255;
                        }
                    }

                    cv::Mat mask_cc;
                    if (disconnect_erode_radius > 0)
                        cv::erode(mask, mask_cc, disconnect_kernel);
                    else
                        mask_cc = mask;

                    cv::Mat labels;
                    int nlabels = cv::connectedComponents(mask_cc, labels, 8, CV_32S);
                    if (nlabels <= 2)
                        return;

                    std::vector<int> counts(nlabels, 0);
                    for (int y = 0; y < labels.rows; ++y) {
                        const int* row = labels.ptr<int>(y);
                        for (int x = 0; x < labels.cols; ++x) {
                            int label = row[x];
                            if (label > 0)
                                counts[label]++;
                        }
                    }

                    int keep_label = 1;
                    for (int label = 2; label < nlabels; ++label)
                        if (counts[label] > counts[keep_label])
                            keep_label = label;

                    cv::Mat keep_mask = (labels == keep_label);
                    if (disconnect_erode_radius > 0)
                        cv::dilate(keep_mask, keep_mask, disconnect_kernel);

                    bool any_removed = false;
                    for (int y = 0; y < labels.rows; ++y) {
                        const uint8_t* keep_row = keep_mask.ptr<uint8_t>(y);
                        for (int x = 0; x < labels.cols; ++x) {
                            if (!mask.at<uint8_t>(y, x))
                                continue;
                            if (keep_row[x])
                                continue;
                            cv::Vec2i p = {y, x_start + x};
                            if ((state(p) & STATE_LOC_VALID) == 0)
                                continue;
                            clear_point(p);
                            removed_total++;
                            any_removed = true;
                            for (const auto& n : neigh8) {
                                cv::Vec2i pn = p + n;
                                if (!point_in_bounds(state, pn))
                                    continue;
                                if (state(pn) & STATE_LOC_VALID)
                                    requeue.insert(pn);
                            }
                        }
                    }
                    if (any_removed)
                        windows_pruned++;
                };

                int last_start = -1;
                for (int x = used_x0; x + win_w <= used_x1; x += stride) {
                    process_window(x);
                    last_start = x;
                }
                int final_start = used_x1 - win_w;
                if (final_start >= used_x0 && final_start != last_start)
                    process_window(final_start);

                if (removed_total > 0) {
                    for (const auto& p : requeue)
                        fringe.insert(p);
                    for(int i=0;i<omp_get_max_threads();i++) {
                        data_ths[i] = data;
                        added_points_threads[i].clear();
                    }
                    for (const auto& p : force_retry)
                        force_retry_mark(p) = 0;
                    force_retry.clear();
                    init_area_scan();
                    std::cout << "disconnect-prune: removed " << removed_total
                              << " points across " << windows_pruned
                              << " windows; requeued " << requeue.size() << std::endl;
                }
            }
        }

        int inl_lower_bound_reg = params.value("consensus_default_th", 10);
        int inl_lower_bound_b = params.value("consensus_limit_th", 2);
        int inl_lower_bound = inl_lower_bound_reg;

        if (!at_right_border && curr_best_inl_th <= inl_lower_bound)
            inl_lower_bound = inl_lower_bound_b;

        if (fringe.empty() && curr_best_inl_th > inl_lower_bound) {
            curr_best_inl_th = std::max(inl_lower_bound, curr_best_inl_th - 1);
            cv::Rect active = active_bounds & used_area;
            for(int j=active.y-2;j<=active.br().y+2;j++)
                for(int i=active.x-2;i<=active.br().x+2;i++)
                    if (state(j,i) & STATE_LOC_VALID)
                            fringe.insert(cv::Vec2i(j,i));
        } else
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

        if (generation % 250 == 0 || update_mapping /*|| generation < 10*/) {
            {
                cv::Mat_<cv::Vec3f> points_hr =
                    surftrack_genpoints_hr(data, state, points, used_area, step, src_step,
                                           /*inpaint=*/false,
                                           /*parallel=*/params.value("hr_gen_parallel", false));
                auto dbg_surf = new QuadSurface(points_hr(used_area_hr), {1/src_step,1/src_step});
                dbg_surf->meta = std::make_unique<nlohmann::json>();
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
                (*dbg_surf->meta)["seed_surface_name"] = seed->id;
                (*dbg_surf->meta)["seed_surface_id"] = seed->id;
                std::string uuid = Z_DBG_GEN_PREFIX+get_surface_time_str();
                dbg_surf->save(tgt_dir / uuid, uuid);
                delete dbg_surf;
            }
        }

        //lets just see what happens
        if (update_mapping) {
            global_opt_count++;
            bool save_inp_hr = (global_opt_count % 5 == 0);
            dbg_counter = generation;
            SurfTrackerData opt_data = data;
            cv::Rect all(0,0,w, h);
            cv::Mat_<uint8_t> opt_state = state.clone();
            cv::Mat_<cv::Vec3d> opt_points = points.clone();
            cv::Mat_<uint16_t> opt_generations = generations.clone();

            cv::Rect active = active_bounds & used_area;
            optimize_surface_mapping(opt_data, opt_state, opt_points, opt_generations, active, static_bounds, step, src_step,
                                     {y0,x0}, closing_r, overlapping_map, patch_index_ptr,
                                     save_inp_hr, tgt_dir,
                                     /*hr_gen_parallel=*/params.value("hr_gen_parallel", false),
                                     /*remap_parallel=*/params.value("remap_parallel", false));
            if (active.area() > 0) {
                copy(opt_data, data, active);
                opt_points(active).copyTo(points(active));
                opt_state(active).copyTo(state(active));
                opt_generations(active).copyTo(generations(active));
                for (int j = active.y; j < active.br().y; ++j)
                    for (int i = active.x; i < active.br().x; ++i) {
                        hole_retry_count(j, i) = 0;
                        force_retry_count(j, i) = 0;
                        force_retry_mark(j, i) = 0;
                    }

                for(int i=0;i<omp_get_max_threads();i++) {
                    data_ths[i] = data;
                    added_points_threads[i].resize(0);
                }
                force_retry.clear();
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
        }

        const double current_area_vx2 = area_accum_vox2;
        const double current_area_cm2 = current_area_vx2 * double(voxelsize) * double(voxelsize) / 1e8;
        if (generation % 100 == 0) {
            printf("gen %d processing %lu fringe cands (total done %d fringe: %lu) area %.0f vx^2 (%f cm^2) best th: %d\n",
                   generation, static_cast<unsigned long>(cands.size()), succ, static_cast<unsigned long>(fringe.size()),
                   current_area_vx2, current_area_cm2, best_inliers_gen);
        }

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

            cv::Mat_<uint8_t> old_hole_retry_count = hole_retry_count;
            hole_retry_count = cv::Mat_<uint8_t>(size, static_cast<uint8_t>(0));
            old_hole_retry_count.copyTo(hole_retry_count(cv::Rect(0,0,old_hole_retry_count.cols,h)));

            cv::Mat_<uint8_t> old_force_retry_count = force_retry_count;
            force_retry_count = cv::Mat_<uint8_t>(size, static_cast<uint8_t>(0));
            old_force_retry_count.copyTo(force_retry_count(cv::Rect(0,0,old_force_retry_count.cols,h)));

            cv::Mat_<uint8_t> old_force_retry_mark = force_retry_mark;
            force_retry_mark = cv::Mat_<uint8_t>(size, static_cast<uint8_t>(0));
            old_force_retry_mark.copyTo(force_retry_mark(cv::Rect(0,0,old_force_retry_mark.cols,h)));

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

        if (generation % 1000 == 0) {
            try {
                writeTiff(tgt_dir / "inliers_sum.tif", inliers_sum_dbg(used_area));
            } catch (const std::exception& ex) {
                std::cerr << "warning: failed to write inliers_sum.tif: " << ex.what() << std::endl;
            }
        }

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
                               /*inpaint=*/false);

    auto surf = new QuadSurface(points_hr(used_area_hr), {1/src_step,1/src_step});

    auto gen_channel = surftrack_generation_channel(generations, used_area, step);
    if (!gen_channel.empty())
        surf->setChannel("generations", gen_channel);

    surf->meta = std::make_unique<nlohmann::json>();
    (*surf->meta)["max_gen"] = surftrack_max_generation(generations, used_area);
    (*surf->meta)["area_vx2"] = area_final_vox2;
    (*surf->meta)["area_cm2"] = area_final_cm2;
    (*surf->meta)["used_approved_segments"] = std::vector<std::string>(used_approved_names.begin(), used_approved_names.end());
    (*surf->meta)["seed_surface_name"] = seed->id;
    (*surf->meta)["seed_surface_id"] = seed->id;

    std::cout << "pointTo total time: " << pointTo_total_ms << " ms" << std::endl;

    return surf;
}
