#pragma once

// Surface tracking helper types and utilities extracted from GrowSurface.cpp
// for better code organization and readability.

#include <opencv2/core.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "ceres/ceres.h"

#include "vc/core/util/Geometry.hpp"  // For loc_valid
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfaceArea.hpp"  // For vc::surface::quadAreaVox2
#include "vc/core/util/SurfaceModeling.hpp"  // For STATE_*, OPTIMIZE_ALL, etc.

namespace vc::surface_helpers {

// ============================================================================
// Deterministic Helpers
// ============================================================================

// SplitMix64 hash function for deterministic randomness
inline uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Deterministic jitter in range [0,1)
inline double det_jitter01(int y, int x, uint64_t salt) {
    uint64_t h = mix64((uint64_t(y) << 32) ^ uint64_t(x) ^ salt);
    const double inv = 1.0 / double(UINT64_C(1) << 53);
    return double(h >> 11) * inv;
}

// Deterministic symmetric jitter in range [-1,1)
inline double det_jitter_symm(int y, int x, uint64_t salt) {
    return 2.0 * det_jitter01(y, x, salt) - 1.0;
}

// ============================================================================
// Geometry Helpers
// ============================================================================

// Bilinear interpolation at fractional position p (row, col) in points matrix
inline cv::Vec3f at_int_inv(const cv::Mat_<cv::Vec3f>& points, cv::Vec2f p) {
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

// Check if point p is within matrix bounds
template <typename T>
inline bool point_in_bounds(const cv::Mat_<T>& mat, const cv::Vec2i& p) {
    return p[0] >= 0 && p[1] >= 0 && p[0] < mat.rows && p[1] < mat.cols;
}

// Try to count a quad (top-left index j,i) if its four corners are STATE_LOC_VALID.
// Returns the quad area (voxel^2) if counted; 0 otherwise.
inline double maybe_quad_area_and_mark(
    int j,
    int i,
    const cv::Mat_<uint8_t>& state,
    const cv::Mat_<cv::Vec3d>& points,
    cv::Mat_<uint8_t>& quad_done) {
    if (j < 0 || i < 0 || j >= quad_done.rows || i >= quad_done.cols)
        return 0.0;
    if (quad_done(j, i))
        return 0.0;
    if ((state(j, i) & STATE_LOC_VALID) && (state(j, i + 1) & STATE_LOC_VALID) &&
        (state(j + 1, i) & STATE_LOC_VALID) &&
        (state(j + 1, i + 1) & STATE_LOC_VALID)) {
        const double a = vc::surface::quadAreaVox2(
            points(j, i), points(j, i + 1), points(j + 1, i), points(j + 1, i + 1));
        quad_done(j, i) = 1;
        return a;
    }
    return 0.0;
}

// ============================================================================
// Comparators
// ============================================================================

// Deterministic ordering for cv::Vec2i (row-major: y, then x)
struct Vec2iLess {
    bool operator()(const cv::Vec2i& a, const cv::Vec2i& b) const {
        return (a[0] < b[0]) || (a[0] == b[0] && a[1] < b[1]);
    }
};

// Deterministic ordering for QuadSurface pointers
struct SurfPtrLess {
    bool operator()(const QuadSurface* a, const QuadSurface* b) const {
        return std::less<const QuadSurface*>()(a, b);
    }
};

// ============================================================================
// Container Types
// ============================================================================

// Sorted vector-based set for QuadSurface pointers
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

    void clear() { _items.clear(); }

    bool empty() const { return _items.empty(); }

    size_t size() const { return _items.size(); }

    iterator begin() { return _items.begin(); }
    iterator end() { return _items.end(); }
    const_iterator begin() const { return _items.begin(); }
    const_iterator end() const { return _items.end(); }

private:
    container_type _items;
};

// ============================================================================
// Residual ID Types
// ============================================================================

using SurfPoint = std::pair<QuadSurface*, cv::Vec2i>;

// Residual block identifier for tracking ceres residuals
class resId_t {
public:
    resId_t() : _type(0), _sm(nullptr) {}

    resId_t(int type, QuadSurface* sm, const cv::Vec2i& p)
        : _type(type), _sm(sm), _p(p) {}

    resId_t(int type, QuadSurface* sm, const cv::Vec2i& a, const cv::Vec2i& b)
        : _type(type), _sm(sm) {
        if (a[0] == b[0]) {
            _p = (a[1] <= b[1]) ? a : b;
        } else {
            _p = (a[0] < b[0]) ? a : b;
        }
    }

    bool operator==(const resId_t& o) const {
        return _type == o._type && _sm == o._sm && _p == o._p;
    }

    int _type;
    QuadSurface* _sm;
    cv::Vec2i _p;
};

// Hash function for resId_t
struct resId_hash {
    size_t operator()(const resId_t& id) const {
        size_t hash1 = std::hash<int>{}(id._type);
        size_t hash2 = std::hash<void*>{}(id._sm);
        size_t hash3 = std::hash<int>{}(id._p[0]);
        size_t hash4 = std::hash<int>{}(id._p[1]);

        // Magic numbers from boost
        size_t hash = hash1 ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
        hash = hash ^ (hash3 + 0x9e3779b9 + (hash << 6) + (hash >> 2));
        hash = hash ^ (hash4 + 0x9e3779b9 + (hash << 6) + (hash >> 2));

        return hash;
    }
};

// Hash function for SurfPoint
struct SurfPoint_hash {
    size_t operator()(const SurfPoint& p) const {
        size_t hash1 = std::hash<void*>{}(p.first);
        size_t hash2 = std::hash<int>{}(p.second[0]);
        size_t hash3 = std::hash<int>{}(p.second[1]);

        // Magic numbers from boost
        size_t hash = hash1 ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
        hash = hash ^ (hash3 + 0x9e3779b9 + (hash << 6) + (hash >> 2));

        return hash;
    }
};

// ============================================================================
// SurfTrackerData Class
// ============================================================================

// Surface tracking data for loss functions
class SurfTrackerData {
public:
    cv::Vec2d& loc(QuadSurface* sm, const cv::Vec2i& loc) { return _data[{sm, loc}]; }

    bool getLoc(QuadSurface* sm, const cv::Vec2i& loc, cv::Vec2d* out) const {
        auto it = _data.find({sm, loc});
        if (it == _data.end())
            return false;
        *out = it->second;
        return true;
    }

    ceres::ResidualBlockId& resId(const resId_t& id) { return _res_blocks[id]; }

    bool hasResId(const resId_t& id) const { return _res_blocks.contains(id); }

    bool has(QuadSurface* sm, const cv::Vec2i& loc) const {
        return _data.contains({sm, loc});
    }

    void erase(QuadSurface* sm, const cv::Vec2i& loc) { _data.erase({sm, loc}); }

    void eraseSurf(QuadSurface* sm, const cv::Vec2i& loc) { _surfs[loc].erase(sm); }

    SurfPtrSet& surfs(const cv::Vec2i& loc) { return _surfs[loc]; }

    const SurfPtrSet& surfsC(const cv::Vec2i& loc) const {
        if (!_surfs.contains(loc))
            return _emptysurfs;
        else
            return _surfs.find(loc)->second;
    }

    cv::Vec3d lookup_int(QuadSurface* sm, const cv::Vec2i& p) {
        auto id = std::make_pair(sm, p);
        if (!_data.contains(id))
            throw std::runtime_error("error, lookup failed!");
        cv::Vec2d l = loc(sm, p);
        if (!loc_valid(sm->rawPoints(), l))
            return {-1, -1, -1};
        return at_int_inv(sm->rawPoints(), l);
    }

    bool valid_int(QuadSurface* sm, const cv::Vec2i& p) {
        auto id = std::make_pair(sm, p);
        if (!_data.contains(id))
            return false;
        cv::Vec2d l = loc(sm, p);
        if (l[0] == -1)
            return false;
        cv::Rect bounds = {0, 0, sm->rawPoints().rows - 2, sm->rawPoints().cols - 2};
        cv::Vec2i li = {static_cast<int>(floor(l[0])), static_cast<int>(floor(l[1]))};
        if (bounds.contains(cv::Point(li))) {
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
        return false;
    }

    static cv::Vec3d lookup_int_loc(QuadSurface* sm, const cv::Vec2f& l) {
        if (!loc_valid(sm->rawPoints(), cv::Vec2d(l[0], l[1])))
            return {-1, -1, -1};
        return at_int_inv(sm->rawPoints(), l);
    }

    void flip_x(int x0) {
        std::cout << " src sizes " << _data.size() << " " << _surfs.size() << std::endl;
        SurfTrackerData old = *this;
        _data.clear();
        _res_blocks.clear();
        _surfs.clear();

        for (auto& it : old._data)
            _data[{it.first.first, {it.first.second[0], x0 + x0 - it.first.second[1]}}] =
                it.second;

        for (auto& it : old._surfs)
            _surfs[{it.first[0], x0 + x0 - it.first[1]}] = it.second;

        std::cout << " flipped sizes " << _data.size() << " " << _surfs.size() << std::endl;
    }

    void clear() {
        _data.clear();
        _res_blocks.clear();
        _surfs.clear();
    }

    std::unordered_map<SurfPoint, cv::Vec2d, SurfPoint_hash> _data;
    std::unordered_map<resId_t, ceres::ResidualBlockId, resId_hash> _res_blocks;
    std::unordered_map<cv::Vec2i, SurfPtrSet> _surfs;
    SurfPtrSet _emptysurfs;
    cv::Vec3d seed_coord;
    cv::Vec2i seed_loc;
};

// ============================================================================
// Loss Function Configuration
// ============================================================================

// Configuration for loss weights (set from JSON via set_loss_params)
struct LossParams {
    float local_cost_inl_th = 0.2f;
    float straight_weight = 0.7f;
    float straight_weight_3D = 4.0f;
    float z_loc_loss_w = 0.1f;
    float dist_loss_2d_w = 1.0f;
    float dist_loss_3d_w = 2.0f;
    float straight_min_count = 1.0f;
};

void set_loss_params(const LossParams& params);
const LossParams& get_loss_params();

// ============================================================================
// Loss Function Helpers
// ============================================================================

// Add 2D distance loss between point p and p+off
int add_surftrack_distloss(
    QuadSurface* sm,
    const cv::Vec2i& p,
    const cv::Vec2i& off,
    SurfTrackerData& data,
    ceres::Problem& problem,
    const cv::Mat_<uint8_t>& state,
    float unit,
    int flags = 0,
    ceres::ResidualBlockId* res = nullptr,
    float w = 1.0f);

// Add 3D distance loss between point p and p+off
int add_surftrack_distloss_3D(
    cv::Mat_<cv::Vec3d>& points,
    const cv::Vec2i& p,
    const cv::Vec2i& off,
    ceres::Problem& problem,
    const cv::Mat_<uint8_t>& state,
    float unit,
    int flags = 0,
    ceres::ResidualBlockId* res = nullptr,
    float w = 2.0f);

// Conditional 2D distance loss (only adds if not already present)
int cond_surftrack_distloss(
    int type,
    QuadSurface* sm,
    const cv::Vec2i& p,
    const cv::Vec2i& off,
    SurfTrackerData& data,
    ceres::Problem& problem,
    const cv::Mat_<uint8_t>& state,
    float unit,
    int flags = 0);

// Conditional 3D distance loss (only adds if not already present)
int cond_surftrack_distloss_3D(
    int type,
    QuadSurface* sm,
    cv::Mat_<cv::Vec3d>& points,
    const cv::Vec2i& p,
    const cv::Vec2i& off,
    SurfTrackerData& data,
    ceres::Problem& problem,
    const cv::Mat_<uint8_t>& state,
    float unit,
    int flags = 0);

// Add 2D straightness loss for three points
int add_surftrack_straightloss(
    QuadSurface* sm,
    const cv::Vec2i& p,
    const cv::Vec2i& o1,
    const cv::Vec2i& o2,
    const cv::Vec2i& o3,
    SurfTrackerData& data,
    ceres::Problem& problem,
    const cv::Mat_<uint8_t>& state,
    int flags = 0,
    float w = 0.7f);

// Add 3D straightness loss for three points
int add_surftrack_straightloss_3D(
    const cv::Vec2i& p,
    const cv::Vec2i& o1,
    const cv::Vec2i& o2,
    const cv::Vec2i& o3,
    cv::Mat_<cv::Vec3d>& points,
    ceres::Problem& problem,
    const cv::Mat_<uint8_t>& state,
    int flags = 0,
    ceres::ResidualBlockId* res = nullptr,
    float w = 4.0f);

// Conditional 3D straightness loss
int cond_surftrack_straightloss_3D(
    int type,
    QuadSurface* sm,
    const cv::Vec2i& p,
    const cv::Vec2i& o1,
    const cv::Vec2i& o2,
    const cv::Vec2i& o3,
    cv::Mat_<cv::Vec3d>& points,
    SurfTrackerData& data,
    ceres::Problem& problem,
    const cv::Mat_<uint8_t>& state,
    int flags = 0);

// Add surface loss
int add_surftrack_surfloss(
    QuadSurface* sm,
    const cv::Vec2i& p,
    SurfTrackerData& data,
    ceres::Problem& problem,
    const cv::Mat_<uint8_t>& state,
    cv::Mat_<cv::Vec3d>& points,
    float step,
    ceres::ResidualBlockId* res = nullptr,
    float w = 0.1f);

// Conditional surface loss
int cond_surftrack_surfloss(
    int type,
    QuadSurface* sm,
    const cv::Vec2i& p,
    SurfTrackerData& data,
    ceres::Problem& problem,
    const cv::Mat_<uint8_t>& state,
    cv::Mat_<cv::Vec3d>& points,
    float step);

// Add local losses for a point (2D and optionally 3D)
int surftrack_add_local(
    QuadSurface* sm,
    const cv::Vec2i& p,
    SurfTrackerData& data,
    ceres::Problem& problem,
    const cv::Mat_<uint8_t>& state,
    cv::Mat_<cv::Vec3d>& points,
    float step,
    float src_step,
    int flags = 0,
    int* straigh_count_ptr = nullptr);

// Add global losses for a point
int surftrack_add_global(
    QuadSurface* sm,
    const cv::Vec2i& p,
    SurfTrackerData& data,
    ceres::Problem& problem,
    const cv::Mat_<uint8_t>& state,
    cv::Mat_<cv::Vec3d>& points,
    float step,
    int flags = 0,
    float step_onsurf = 0);

// ============================================================================
// Local Cost Evaluation
// ============================================================================

// Radius for local cost patch
int local_cost_patch_radius();

// Compute local cost destructively (modifies and restores state)
double local_cost_destructive(
    QuadSurface* sm,
    const cv::Vec2i& p,
    SurfTrackerData& data,
    cv::Mat_<uint8_t>& state,
    cv::Mat_<cv::Vec3d>& points,
    float step,
    float src_step,
    cv::Vec3f loc,
    int* ref_count = nullptr,
    int* straight_count_ptr = nullptr);

// Compute local cost from snapshot (thread-safe)
double local_cost_snapshot(
    QuadSurface* sm,
    const cv::Vec2i& p,
    const SurfTrackerData& data_src,
    const cv::Mat_<uint8_t>& state_src,
    const cv::Mat_<cv::Vec3d>& points_src,
    float step,
    float src_step,
    const cv::Vec3f& loc,
    std::shared_mutex& mutex,
    int* ref_count = nullptr,
    int* straight_count_ptr = nullptr);

// Compute local cost (read-only)
double local_cost(
    QuadSurface* sm,
    const cv::Vec2i& p,
    SurfTrackerData& data,
    const cv::Mat_<uint8_t>& state,
    cv::Mat_<cv::Vec3d>& points,
    float step,
    float src_step,
    int* ref_count = nullptr,
    int* straight_count_ptr = nullptr);

// Local solve with ceres
double local_solve(
    QuadSurface* sm,
    const cv::Vec2i& p,
    SurfTrackerData& data,
    const cv::Mat_<uint8_t>& state,
    cv::Mat_<cv::Vec3d>& points,
    float step,
    float src_step,
    int flags);

// ============================================================================
// Generation/HR Output
// ============================================================================

// Round step to integer
int surftrack_round_step(float step);

// Generate generation channel at higher resolution
cv::Mat_<uint16_t> surftrack_generation_channel(
    const cv::Mat_<uint16_t>& generations,
    const cv::Rect& used_area,
    float step);

// Get max generation in area
uint16_t surftrack_max_generation(
    const cv::Mat_<uint16_t>& generations,
    const cv::Rect& used_area);

// Generate high-resolution points from surface tracking data
cv::Mat_<cv::Vec3f> surftrack_genpoints_hr(
    SurfTrackerData& data,
    cv::Mat_<uint8_t>& state,
    cv::Mat_<cv::Vec3d>& points,
    const cv::Rect& used_area,
    float step,
    float step_src,
    bool inpaint = false,
    bool parallel = true);

// Thin-plate inpainting for hole filling
void thin_plate_inpaint(
    cv::Mat_<cv::Vec3d>& points,
    const cv::Mat_<uint8_t>& state,
    const cv::Mat_<uint8_t>& inpaint_mask,
    const cv::Rect& used_area,
    int warmup_iters,
    int thin_plate_iters);

}  // namespace vc::surface_helpers
