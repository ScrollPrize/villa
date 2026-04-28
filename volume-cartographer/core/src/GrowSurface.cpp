#include <omp.h>


#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/core/util/Umbilicus.hpp"
#include "vc/tracer/SurfaceModeling.hpp"
#include "vc/core/util/OMPThreadPointCollection.hpp"
#include "vc/core/util/DateTime.hpp"
#include "vc/core/util/HashFunctions.hpp"
#include "vc/core/types/ChunkedTensor.hpp"

#include "vc/core/util/LifeTime.hpp"
#include "vc/tracer/Tracer.hpp"
#include "utils/Json.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fstream>
#include <iostream>

using vec2i_hash = std::vec2i_hash;

namespace {

static nlohmann::json json_from_utils(const utils::Json& json)
{
    return nlohmann::json::parse(json.dump());
}

static utils::Json json_to_utils(const nlohmann::json& json)
{
    return utils::Json::parse(json.dump());
}

static cv::Rect scaled_rect_trunc(const cv::Rect& rect, float scale)
{
    return {
        static_cast<int>(rect.x * scale),
        static_cast<int>(rect.y * scale),
        static_cast<int>(rect.width * scale),
        static_cast<int>(rect.height * scale)
    };
}

static cv::Rect valid_points_bounds(const cv::Mat_<cv::Vec3d>& points)
{
    cv::Rect bounds;
    bool have_valid = false;
    for (int y = 0; y < points.rows; ++y) {
        for (int x = 0; x < points.cols; ++x) {
            if (points(y, x)[0] == -1) {
                continue;
            }
            const cv::Rect cell(x, y, 1, 1);
            bounds = have_valid ? (bounds | cell) : cell;
            have_valid = true;
        }
    }
    return bounds;
}

static std::string surface_name(const QuadSurface* surf)
{
    if (!surf)
        return {};
    if (!surf->path.empty() && !surf->path.filename().empty())
        return surf->path.filename().string();
    return surf->id;
}

static bool looks_like_resume_surface(const QuadSurface* surf, const nlohmann::json& params)
{
    if (params.value("resume_growth", false) || params.value("resume", false)) {
        return true;
    }
    if (!surf || surf->meta.is_null() || !surf->meta.is_object()) {
        return false;
    }

    const nlohmann::json meta = json_from_utils(surf->meta);
    if (meta.contains("vc_grow_seg_from_segments_params")) {
        return true;
    }
    if (meta.value("source", std::string{}) == "vc_grow_seg_from_segments") {
        return true;
    }
    return false;
}

using SurfaceOverlaps = std::unordered_map<QuadSurface*, std::set<QuadSurface*>>;
using Umbilicus = vc::core::util::Umbilicus;

static const std::set<QuadSurface*>& surface_overlaps(const SurfaceOverlaps* overlaps, QuadSurface* surf)
{
    static const std::set<QuadSurface*> empty;
    if (!overlaps)
        return empty;
    auto it = overlaps->find(surf);
    return it == overlaps->end() ? empty : it->second;
}

static cv::Vec3i volume_shape_from_params(const nlohmann::json& params)
{
    if (!params.contains("_volume_shape") || !params["_volume_shape"].is_array() || params["_volume_shape"].size() != 3) {
        throw std::runtime_error("\"single wrap\" requires internal _volume_shape [z, y, x]; command callers should derive it from the volume path");
    }

    return {
        params["_volume_shape"][0].get<int>(),
        params["_volume_shape"][1].get<int>(),
        params["_volume_shape"][2].get<int>()
    };
}

static std::optional<Umbilicus> single_wrap_umbilicus_from_params(const nlohmann::json& params)
{
    if (!params.value("single wrap", false)) {
        return std::nullopt;
    }
    if (!params.contains("umbilicus_path") || !params["umbilicus_path"].is_string()) {
        throw std::runtime_error("\"single wrap\" requires an umbilicus_path parameter");
    }

    Umbilicus umbilicus = Umbilicus::FromFile(params["umbilicus_path"].get<std::string>(),
                                              volume_shape_from_params(params));
    umbilicus.set_seam(Umbilicus::SeamDirection::NegativeX);
    return umbilicus;
}

static bool crosses_single_wrap_seam(const Umbilicus& umbilicus,
                                     const cv::Vec3d& a,
                                     const cv::Vec3d& b)
{
    const int za = static_cast<int>(std::lround(a[2]));
    const int zb = static_cast<int>(std::lround(b[2]));
    if (za != zb) {
        return false;
    }

    const cv::Vec3f af{static_cast<float>(a[0]), static_cast<float>(a[1]), static_cast<float>(a[2])};
    const cv::Vec3f bf{static_cast<float>(b[0]), static_cast<float>(b[1]), static_cast<float>(b[2])};
    const double theta_a = umbilicus.theta(af);
    const double theta_b = umbilicus.theta(bf);
    double delta = std::abs(theta_a - theta_b);
    if (delta > 360.0) {
        delta = std::fmod(delta, 360.0);
    }
    return delta > 180.0;
}

static bool same_single_wrap_z(const cv::Vec3d& a, const cv::Vec3d& b)
{
    return static_cast<int>(std::lround(a[2])) == static_cast<int>(std::lround(b[2]));
}

static double angular_distance_degrees(double a, double b)
{
    double delta = std::abs(a - b);
    if (delta > 360.0) {
        delta = std::fmod(delta, 360.0);
    }
    return std::min(delta, 360.0 - delta);
}

static cv::Vec3f vec3d_to_vec3f(const cv::Vec3d& p)
{
    return {static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2])};
}

} // namespace

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
static int sdir_3d_radius = 2;             // Symmetric-Dirichlet metric radius, in grid strides
static float sdir_3d_w = 0.5f;             // Weight for local 3D metric preservation
static float sdir_3d_global_w = 0.25f;     // Weight for 3D metric preservation during global optimization
static float sdir_3d_candidate_max = 4.0f; // Reject candidates with a worse local metric residual
static float straight_min_count = 1.0f;    // Minimum number of straight constraints
static int inlier_base_threshold = 20;     // Starting threshold for inliers

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

using SurfPoint = std::pair<QuadSurface *,cv::Vec2i>;

class resId_t
{
public:
    resId_t() : _type(0), _sm(nullptr) {
    } ;
    resId_t(int type, QuadSurface * sm, const cv::Vec2i& p) : _type(type), _sm(sm), _p(p) {};
    resId_t(int type, QuadSurface * sm, const cv::Vec2i &a, const cv::Vec2i &b) : _type(type), _sm(sm)
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
    QuadSurface * _sm;
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
    std::set<QuadSurface *> &surfs(const cv::Vec2i &loc)
    {
        return _surfs[loc];
    }
    const std::set<QuadSurface *> &surfsC(const cv::Vec2i &loc) const
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
        if (l[0] == -1)
            return {-1,-1,-1};
        else {
            cv::Rect bounds = {0, 0, sm->rawPoints().rows-2,sm->rawPoints().cols-2};
            cv::Vec2i li = {floor(l[0]),floor(l[1])};
            if (bounds.contains(cv::Point(li)))
                return at_int_inv(sm->rawPoints(), l);
            else
                return {-1,-1,-1};
        }
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
            cv::Vec2i li = {floor(l[0]),floor(l[1])};
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
        if (l[0] == -1)
            return {-1,-1,-1};
        else {
            cv::Rect bounds = {0, 0, sm->rawPoints().rows-2,sm->rawPoints().cols-2};
            if (bounds.contains(cv::Point(l)))
                return at_int_inv(sm->rawPoints(), l);
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

        for(auto &it : old._data)
            _data[{it.first.first,{it.first.second[0],x0+x0-it.first.second[1]}}] = it.second;

        for(auto &it : old._surfs)
            _surfs[{it.first[0],x0+x0-it.first[1]}] = it.second;

        std::cout << " flipped sizes " << _data.size() << " " << _surfs.size() << std::endl;
    }
    void translate(const cv::Vec2i& delta)
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
// protected:
    std::unordered_map<SurfPoint,cv::Vec2d,SurfPoint_hash> _data;
    std::unordered_map<resId_t,ceres::ResidualBlockId,resId_hash> _res_blocks;
    std::unordered_map<cv::Vec2i,std::set<QuadSurface *>,vec2i_hash> _surfs;
    std::set<QuadSurface *> _emptysurfs;
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

    for(auto &it : src._data)
        if (roi.contains(cv::Point(it.first.second)))
            tgt._data[it.first] = it.second;
    for(auto &it : src._surfs)
        if (roi.contains(cv::Point(it.first)))
            tgt._surfs[it.first] = it.second;

    // tgt.seed_loc = src.seed_loc;
    // tgt.seed_coord = src.seed_coord;
}

static void add_surface_if_close(QuadSurface* surf,
                                 const cv::Vec2i& p,
                                 const cv::Vec3d& coord,
                                 SurfTrackerData& data,
                                 SurfacePatchIndex* surface_patch_index)
{
    if (!surf || data.has(surf, p)) {
        return;
    }

    auto ptr = surf->pointer();
    const cv::Vec3f coord_f{
        static_cast<float>(coord[0]),
        static_cast<float>(coord[1]),
        static_cast<float>(coord[2])
    };

    if (surface_patch_index && !surface_patch_index->empty()) {
        SurfacePatchIndex::SurfacePtr target_surface(surf, [](QuadSurface*) {});
        auto hit = surface_patch_index->locate(coord_f, same_surface_th, target_surface);
        if (!hit) {
            return;
        }
        ptr = hit->ptr;
    } else {
        if (surf->pointTo(ptr, coord_f, same_surface_th, 10) > same_surface_th) {
            return;
        }
    }

    const cv::Vec3f loc = surf->loc_raw(ptr);
    if (SurfTrackerData::lookup_int_loc(surf, {loc[1], loc[0]})[0] == -1) {
        return;
    }
    data.surfs(p).insert(surf);
    data.loc(surf, p) = {loc[1], loc[0]};
}

static int add_surftrack_distloss(QuadSurface *sm, const cv::Vec2i &p, const cv::Vec2i &off, SurfTrackerData &data,
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

static int add_surftrack_sdirloss_3D(cv::Mat_<cv::Vec3d> &points, const cv::Vec2i &p, int stride,
    ceres::Problem &problem, const cv::Mat_<uint8_t> &state, float unit, float weight = sdir_3d_w)
{
    if (weight <= 0.0f || stride < 1)
        return 0;

    const cv::Vec2i pu = p + cv::Vec2i(0, stride);
    const cv::Vec2i pv = p + cv::Vec2i(stride, 0);
    if (p[0] < 0 || p[0] >= state.rows || p[1] < 0 || p[1] >= state.cols)
        return 0;
    if (pu[0] < 0 || pu[0] >= state.rows || pu[1] < 0 || pu[1] >= state.cols)
        return 0;
    if (pv[0] < 0 || pv[0] >= state.rows || pv[1] < 0 || pv[1] >= state.cols)
        return 0;
    if ((state(p) & (STATE_COORD_VALID | STATE_LOC_VALID)) == 0)
        return 0;
    if ((state(pu) & (STATE_COORD_VALID | STATE_LOC_VALID)) == 0)
        return 0;
    if ((state(pv) & (STATE_COORD_VALID | STATE_LOC_VALID)) == 0)
        return 0;
    if (points(p)[0] == -1 || points(pu)[0] == -1 || points(pv)[0] == -1)
        return 0;

    problem.AddResidualBlock(
        SymmetricDirichletLoss::Create(unit * stride, weight, 1e-8, 1e-2),
        new ceres::CauchyLoss(1.0),
        &points(p)[0],
        &points(pu)[0],
        &points(pv)[0]);

    return 1;
}

static int add_surftrack_sdirlosses_3D(cv::Mat_<cv::Vec3d> &points, const cv::Vec2i &p, ceres::Problem &problem,
    const cv::Mat_<uint8_t> &state, float unit)
{
    if (sdir_3d_radius < 1 || sdir_3d_w <= 0.0f)
        return 0;

    int count = 0;
    for (int stride = 1; stride <= sdir_3d_radius; ++stride) {
        count += add_surftrack_sdirloss_3D(points, p, stride, problem, state, unit);
        count += add_surftrack_sdirloss_3D(points, p - cv::Vec2i(0, stride), stride, problem, state, unit);
        count += add_surftrack_sdirloss_3D(points, p - cv::Vec2i(stride, 0), stride, problem, state, unit);
    }

    return count;
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

static int cond_surftrack_sdirloss_3D(int type, QuadSurface *sm, cv::Mat_<cv::Vec3d> &points, const cv::Vec2i &p, int stride,
    SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state, float unit)
{
    resId_t id(type, sm, p);
    if (data.hasResId(id))
        return 0;

    const int count = add_surftrack_sdirloss_3D(points, p, stride, problem, state, unit, sdir_3d_global_w);
    if (count) {
        data.resId(id) = nullptr;
    }

    return count;
}

static int cond_surftrack_sdirlosses_3D(QuadSurface *sm, cv::Mat_<cv::Vec3d> &points, const cv::Vec2i &p,
    SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state, float unit)
{
    if (sdir_3d_radius < 1 || sdir_3d_global_w <= 0.0f)
        return 0;

    int count = 0;
    int type = 20;
    for (int stride = 1; stride <= sdir_3d_radius; ++stride) {
        count += cond_surftrack_sdirloss_3D(type++, sm, points, p, stride, data, problem, state, unit);
        count += cond_surftrack_sdirloss_3D(type++, sm, points, p - cv::Vec2i(0, stride), stride, data, problem, state, unit);
        count += cond_surftrack_sdirloss_3D(type++, sm, points, p - cv::Vec2i(stride, 0), stride, data, problem, state, unit);
    }

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
static int surftrack_add_local(QuadSurface *sm, const cv::Vec2i& p, SurfTrackerData &data, ceres::Problem &problem,
    const cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points, float step, float src_step, int flags = 0, int *straigh_count_ptr = nullptr)
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
        count += add_surftrack_sdirlosses_3D(points, p, problem, state, step*src_step);

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
        count += cond_surftrack_sdirlosses_3D(sm, points, p, data, problem, state, step);

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


static bool candidate_coord_at(const cv::Vec2i& candidate_p, const cv::Vec3d& candidate_coord,
    const cv::Vec2i& p, const cv::Mat_<uint8_t>& state, const cv::Mat_<cv::Vec3d>& points, cv::Vec3d& out)
{
    if (p[0] < 0 || p[0] >= state.rows || p[1] < 0 || p[1] >= state.cols)
        return false;
    if (p == candidate_p) {
        out = candidate_coord;
        return out[0] != -1;
    }
    if ((state(p) & (STATE_LOC_VALID | STATE_COORD_VALID)) == 0)
        return false;
    if (points(p)[0] == -1)
        return false;

    out = points(p);
    return true;
}

static bool candidate_sdir_residual(const cv::Vec2i& candidate_p, const cv::Vec3d& candidate_coord,
    const cv::Vec2i& origin, int stride, const cv::Mat_<uint8_t>& state, const cv::Mat_<cv::Vec3d>& points,
    float unit, double& residual)
{
    cv::Vec3d p;
    cv::Vec3d pu;
    cv::Vec3d pv;
    if (!candidate_coord_at(candidate_p, candidate_coord, origin, state, points, p))
        return false;
    if (!candidate_coord_at(candidate_p, candidate_coord, origin + cv::Vec2i(0, stride), state, points, pu))
        return false;
    if (!candidate_coord_at(candidate_p, candidate_coord, origin + cv::Vec2i(stride, 0), state, points, pv))
        return false;

    const double scaled_unit = static_cast<double>(unit) * static_cast<double>(stride);
    if (scaled_unit <= 0.0)
        return false;

    const cv::Vec3d eu = (pu - p) / scaled_unit;
    const cv::Vec3d ev = (pv - p) / scaled_unit;

    const double a = eu.dot(eu);
    const double c = ev.dot(ev);
    const double b = eu.dot(ev);
    const double trG = a + c;
    const double detG = a * c - b * b;
    const double det_safe = detG + 1e-8 + 1e-2 * trG;
    if (det_safe <= 0.0)
        return false;

    const double energy = trG + trG / det_safe;
    residual = std::abs(energy - 4.0);
    return true;
}

static bool local_metric_consistent(const cv::Vec2i& p, const cv::Vec3d& coord,
    const cv::Mat_<uint8_t>& state, const cv::Mat_<cv::Vec3d>& points, float step, float src_step)
{
    if (sdir_3d_radius < 1 || sdir_3d_candidate_max <= 0.0f)
        return true;

    const float unit = step * src_step;
    for (int stride = 1; stride <= sdir_3d_radius; ++stride) {
        const cv::Vec2i origins[] = {
            p,
            p - cv::Vec2i(0, stride),
            p - cv::Vec2i(stride, 0)
        };
        for (const cv::Vec2i& origin : origins) {
            double residual = 0.0;
            if (!candidate_sdir_residual(p, coord, origin, stride, state, points, unit, residual))
                continue;
            if (residual > static_cast<double>(sdir_3d_candidate_max))
                return false;
        }
    }

    return true;
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


static cv::Mat_<cv::Vec3d> surftrack_genpoints_hr(SurfTrackerData &data, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points, const cv::Rect &used_area,
    float step, float step_src, bool inpaint = false)
{
    cv::Mat_<cv::Vec3f> points_hr(state.rows*step, state.cols*step, {0,0,0});
    cv::Mat_<int> counts_hr(state.rows*step, state.cols*step, 0);
#pragma omp parallel for //FIXME data access is just not threading friendly ...
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
                                points_hr(j*step+sy,i*step+sx) += SurfTrackerData::lookup_int_loc(sm,l);
                                counts_hr(j*step+sy,i*step+sx) += 1;
                            }
                        }
                }
            }
            if (!counts_hr(j*step+1,i*step+1) && inpaint) {
                const cv::Vec3d& c00 = points(j,i);
                const cv::Vec3d& c01 = points(j,i+1);
                const cv::Vec3d& c10 = points(j+1,i);
                const cv::Vec3d& c11 = points(j+1,i+1);

                for(int sy=0;sy<=step;sy++)
                    for(int sx=0;sx<=step;sx++) {
                        if (!counts_hr(j*step+sy,i*step+sx)) {
                            float fx = sx/step;
                            float fy = sy/step;
                            cv::Vec3d c0 = (1-fx)*c00 + fx*c01;
                            cv::Vec3d c1 = (1-fx)*c10 + fx*c11;
                            cv::Vec3d c = (1-fy)*c0 + fy*c1;
                            points_hr(j*step+sy,i*step+sx) = c;
                            counts_hr(j*step+sy,i*step+sx) = 1;
                        }
                    }
            }
        }
    }
#pragma omp parallel for
    for(int j=0;j<points_hr.rows;j++)
        for(int i=0;i<points_hr.cols;i++)
            if (counts_hr(j,i))
                points_hr(j,i) /= counts_hr(j,i);
            else
                points_hr(j,i) = {-1,-1,-1};

    return points_hr;
}

static cv::Mat_<uint16_t> surftrack_generations_hr(const cv::Mat_<uint8_t>& state, const cv::Mat_<uint16_t>& generations,
                                                   const cv::Rect& used_area, float step)
{
    cv::Mat_<uint16_t> generations_hr(state.rows * step, state.cols * step, static_cast<uint16_t>(0));
    for(int j=used_area.y;j<used_area.br().y-1;j++)
        for(int i=used_area.x;i<used_area.br().x-1;i++) {
            if (state(j,i) & (STATE_LOC_VALID|STATE_COORD_VALID)
                && state(j,i+1) & (STATE_LOC_VALID|STATE_COORD_VALID)
                && state(j+1,i) & (STATE_LOC_VALID|STATE_COORD_VALID)
                && state(j+1,i+1) & (STATE_LOC_VALID|STATE_COORD_VALID))
            {
                uint16_t cell_generation = std::max(
                    std::max(generations(j,i), generations(j,i+1)),
                    std::max(generations(j+1,i), generations(j+1,i+1)));
                if (cell_generation == 0) {
                    continue;
                }
                for(int sy=0;sy<=step;sy++)
                    for(int sx=0;sx<=step;sx++) {
                        const int y = j * step + sy;
                        const int x = i * step + sx;
                        generations_hr(y,x) = std::max(generations_hr(y,x), cell_generation);
                    }
            }
        }

    return generations_hr;
}




//try flattening the current surface mapping assuming direct 3d distances
//this is basically just a reparametrization
static void optimize_surface_mapping(SurfTrackerData &data, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points, cv::Rect used_area,
    cv::Rect static_bounds, float step, float src_step, const cv::Vec2i &seed, int closing_r, bool keep_inpainted = false,
    const std::filesystem::path& tgt_dir = std::filesystem::path(),
    SurfacePatchIndex* surface_patch_index = nullptr,
    const SurfaceOverlaps* overlaps = nullptr,
    bool debug_images = false)
{
    std::cout << "optimizer: optimizing surface " << state.size() << " " << used_area <<  " " << static_bounds << std::endl;

    cv::Mat_<cv::Vec3d> points_new = points.clone();
    // QuadSurface destruction can release storage shared with optimizer helper
    // surfaces on this path. Keep these remap-only helpers alive past the
    // optimizer cleanup.
    QuadSurface* sm = new QuadSurface(points, {1,1});

    std::shared_mutex mutex;

    SurfTrackerData data_new;
    data_new._data = data._data;

    used_area = cv::Rect(used_area.x-2,used_area.y-2,used_area.size().width+4,used_area.size().height+4);
    cv::Rect used_area_hr = scaled_rect_trunc(used_area, step);

    ceres::Problem problem_inpaint;
    ceres::Solver::Summary summary;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
#ifdef VC_USE_CUDA_SPARSE
    // Check if Ceres was actually built with CUDA sparse support
    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::CUDA_SPARSE)) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;

    } else {
        std::cerr << "Warning: CUDA_SPARSE requested but Ceres was not built with CUDA sparse support. Falling back to default solver." << std::endl;
    }
#endif
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 100;
    options.num_threads = omp_get_max_threads();

    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++)
            if (state(j,i) & STATE_LOC_VALID) {
                data_new.surfs({j,i}).insert(sm);
                data_new.loc(sm, {j,i}) = {static_cast<double>(j), static_cast<double>(i)};
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
                    double err = local_solve(sm, {j,i}, data_new, new_state, points_new, step, src_step, LOSS_3D_INDIRECT | SURF_LOSS);
                    if (points_new(j,i)[0] == -3) {
                        //FIXME actually check for solver failure?
                        new_state(j, i) = 0;
                        points_new(j,i) = {-1,-1,-1};
                    }
                    else
                        res_count += surftrack_add_global(sm, {j,i}, data_new, problem_inpaint, new_state, points_new, step*src_step, LOSS_3D_INDIRECT | OPTIMIZE_ALL);
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
    QuadSurface* sm_inp = new QuadSurface(points_inpainted, {1,1});

    SurfTrackerData data_inp;
    data_inp._data = data_new._data;

    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++)
            if (new_state(j,i) & STATE_LOC_VALID) {
                data_inp.surfs({j,i}).insert(sm_inp);
                data_inp.loc(sm_inp, {j,i}) = {static_cast<double>(j), static_cast<double>(i)};
            }

    ceres::Problem problem;

    std::cout << "optimizer: using " << used_area.tl() << used_area.br() << std::endl;

    int fix_points = 0;
    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++) {
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
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
    std::cout << "optimizer: rms " << sqrt(summary.final_cost/summary.num_residual_blocks) << " count " << summary.num_residual_blocks << std::endl;

    if (debug_images) {
        cv::Mat_<cv::Vec3d> points_hr_inp = surftrack_genpoints_hr(data, new_state, points_inpainted, used_area, step, src_step, true);
        try {
            auto dbg_surf = new QuadSurface(points_hr_inp(used_area_hr), {1/src_step,1/src_step});
            std::string uuid = std::string(Z_DBG_GEN_PREFIX)+"inp_hr";
            dbg_surf->save(tgt_dir / uuid, uuid, true);
            delete dbg_surf;
        } catch (cv::Exception&) {
            // We did not find a valid region of interest to expand to
            std::cout << "optimizer: no valid region of interest found" << std::endl;
        }
    }

    cv::Mat_<cv::Vec3d> points_hr = surftrack_genpoints_hr(data, new_state, points_inpainted, used_area, step, src_step);
    SurfTrackerData data_out;
    cv::Mat_<cv::Vec3d> points_out(points.size(), {-1,-1,-1});
    cv::Mat_<uint8_t> state_out(state.size(), 0);
    cv::Mat_<uint8_t> support_count(state.size(), 0);
#pragma omp parallel for
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
                int y = l[0];
                int x = l[1];
                l *= step;
                if (loc_valid(points_hr, l)) {
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

                    std::set<QuadSurface *> surfs;
                    surfs.insert(data.surfsC({y,x}).begin(), data.surfsC({y,x}).end());
                    surfs.insert(data.surfsC({y,x+1}).begin(), data.surfsC({y,x+1}).end());
                    surfs.insert(data.surfsC({y+1,x}).begin(), data.surfsC({y+1,x}).end());
                    surfs.insert(data.surfsC({y+1,x+1}).begin(), data.surfsC({y+1,x+1}).end());

                    for(auto &s : surfs) {
                        auto ptr = s->pointer();
                        float res = s->pointTo(ptr, points_out(j, i), same_surface_th, 10,
                                                           surface_patch_index);
                        if (res <= same_surface_th) {
                            mutex.lock();
                            data_out.surfs({j,i}).insert(s);
                            cv::Vec3f loc = s->loc_raw(ptr);
                            data_out.loc(s, {j,i}) = {loc[1], loc[0]};
                            mutex.unlock();
                        }
                    }
                }
            }

    //now filter by consistency
    for(int j=used_area.y;j<used_area.br().y-1;j++)
        for(int i=used_area.x;i<used_area.br().x-1;i++)
            if (!static_bounds.contains(cv::Point(i,j)) && state_out(j,i) & STATE_VALID) {
                std::set<QuadSurface *> surf_src = data_out.surfs({j,i});
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
                    std::set<QuadSurface *> surf_cands = data_out.surfs({j,i});
                    for(auto s : data_out.surfs({j,i}))
                        surf_cands.insert(surface_overlaps(overlaps, s).begin(),
                                          surface_overlaps(overlaps, s).end());
                    mutex.unlock();

                    for(auto test_surf : surf_cands) {
                        mutex.lock_shared();
                        if (data_out.has(test_surf, {j,i})) {
                            mutex.unlock();
                            continue;
                        }
                        mutex.unlock();

                        auto ptr = test_surf->pointer();
                        if (test_surf->pointTo(ptr, points_out(j, i), same_surface_th, 10,
                                                           surface_patch_index) > same_surface_th)
                            continue;

                        int count = 0;
                        cv::Vec3f loc_3d = test_surf->loc_raw(ptr);
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

    if (debug_images) {
        cv::Mat_<cv::Vec3d> points_hr_inp = surftrack_genpoints_hr(data, state, points, used_area, step, src_step, true);
        try {
            auto dbg_surf = new QuadSurface(points_hr_inp(used_area_hr), {1/src_step,1/src_step});
            std::string uuid = std::string(Z_DBG_GEN_PREFIX)+"opt_inp_hr";
            dbg_surf->save(tgt_dir / uuid, uuid, true);
            delete dbg_surf;
        } catch (cv::Exception&) {
            // We did not find a valid region of interest to expand to
            std::cout << "optimizer: no valid region of interest found" << std::endl;
        }
    }

    dbg_counter++;
}

static QuadSurface *grow_surf_from_surfs_impl(QuadSurface *seed,
                                              const std::vector<QuadSurface *> &surfs_v,
                                              const nlohmann::json &params,
                                              float voxelsize,
                                              SurfacePatchIndex* external_surface_patch_index = nullptr)
{
    bool flip_x = params.value("flip_x", 0);
    int global_steps_per_window = params.value("global_steps_per_window", 0);


    std::cout << "global_steps_per_window: " << global_steps_per_window << std::endl;
    std::cout << "flip_x: " << flip_x << std::endl;
    std::filesystem::path tgt_dir = params["tgt_dir"].get<std::string>();

    std::unordered_map<std::string,QuadSurface *> surfs;
    SurfaceOverlaps overlaps;
    float src_step = params.value("src_step", 20);
    float step = params.value("step", 10);
    int max_width = params.value("max_width", 80000);
    const bool use_patch_cache = params.value("use_patch_cache", false);
    const bool debug_images = params.value("debug_images", false);
    std::filesystem::path surface_patch_cache_dir = tgt_dir / ".surface_patch_index_cache";
    if (params.contains("surface_patch_cache_dir") && params["surface_patch_cache_dir"].is_string()) {
        surface_patch_cache_dir = params["surface_patch_cache_dir"].get<std::string>();
    }

    local_cost_inl_th = params.value("local_cost_inl_th", 0.2f);
    same_surface_th = params.value("same_surface_th", 2.0f);
    straight_weight = params.value("straight_weight", 0.7f);            // Weight for 2D straight line constraints
    straight_weight_3D = params.value("straight_weight_3D", 4.0f);      // Weight for 3D straight line constraints
    sliding_w_scale = params.value("sliding_w_scale", 1.0f);            // Scale factor for sliding window
    z_loc_loss_w = params.value("z_loc_loss_w", 0.1f);                  // Weight for Z location loss constraints
    dist_loss_2d_w = params.value("dist_loss_2d_w", 1.0f);              // Weight for 2D distance constraints
    dist_loss_3d_w = params.value("dist_loss_3d_w", 2.0f);              // Weight for 3D distance constraints
    sdir_3d_radius = params.value("sdir_3d_radius", 2);
    sdir_3d_w = params.value("sdir_3d_w", 0.5f);
    sdir_3d_global_w = params.value("sdir_3d_global_w", 0.5f * sdir_3d_w);
    sdir_3d_candidate_max = params.value("sdir_3d_candidate_max", 4.0f);
    straight_min_count = params.value("straight_min_count", 1.0f);      // Minimum number of straight constraints
    inlier_base_threshold = params.value("inlier_base_threshold", 20);  // Starting threshold for inliers

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
    std::cout << "  sdir_3d_radius: " << sdir_3d_radius << std::endl;
    std::cout << "  sdir_3d_w: " << sdir_3d_w << std::endl;
    std::cout << "  sdir_3d_global_w: " << sdir_3d_global_w << std::endl;
    std::cout << "  sdir_3d_candidate_max: " << sdir_3d_candidate_max << std::endl;
    std::cout << "  use_patch_cache: "
              << (use_patch_cache ? "true" : "false") << std::endl;
    if (external_surface_patch_index) {
        std::cout << "  external_surface_patch_index: true" << std::endl;
    }
    if (use_patch_cache) {
        std::cout << "  surface_patch_read_only: true" << std::endl;
        std::cout << "  surface_patch_cache_dir: " << surface_patch_cache_dir << std::endl;
    }
    if (enforce_z_range)
        std::cout << "  z_range: [" << z_min << ", " << z_max << "]" << std::endl;

    std::optional<Umbilicus> single_wrap_umbilicus = single_wrap_umbilicus_from_params(params);
    if (single_wrap_umbilicus) {
        std::cout << "  single wrap: true, seam direction: -x, umbilicus_path: "
                  << params["umbilicus_path"].get<std::string>() << std::endl;
    }
    const int single_wrap_gap_search = std::max(0, params.value("single_wrap_gap_search", 0));
    const double single_wrap_ray_angle_deg = std::max(0.0, params.value("single_wrap_ray_angle_deg", 2.0));
    const double single_wrap_ray_z_tolerance = std::max(0.0, params.value("single_wrap_ray_z_tolerance", 1.0));
    const double single_wrap_ray_radius_tolerance = std::max(
        0.0,
        params.value("single_wrap_ray_radius_tolerance",
                     static_cast<double>(std::max(src_step * step, 1.0f)) * 0.75));
    if (single_wrap_umbilicus) {
        std::cout << "  single_wrap_gap_search: "
                  << (single_wrap_gap_search == 0 ? std::string("used-area") : std::to_string(single_wrap_gap_search))
                  << std::endl;
        std::cout << "  single_wrap_ray_angle_deg: " << single_wrap_ray_angle_deg
                  << ", single_wrap_ray_z_tolerance: " << single_wrap_ray_z_tolerance
                  << ", single_wrap_ray_radius_tolerance: " << single_wrap_ray_radius_tolerance
                  << std::endl;
    }

    std::cout << "total surface count: " << surfs_v.size() << std::endl;

    std::set<QuadSurface *> approved_sm;

    std::set<std::string> used_approved_names;
    std::string log_filename = "/tmp/vc_grow_seg_from_segments_" + get_surface_time_str() + "_used_approved_segments.txt";
    std::ofstream approved_log(log_filename);

    for(auto &sm : surfs_v) {
        const nlohmann::json meta = json_from_utils(sm->meta);
        if (meta.contains("tags") && meta.at("tags").contains("approved"))
            approved_sm.insert(sm);
        if (!meta.contains("tags") || !meta.at("tags").contains("defective")) {
            surfs[surface_name(sm)] = sm;
        }
    }

    for(auto sm : approved_sm)
        std::cout << "approved: " << surface_name(sm) << std::endl;

    for(auto &sm : surfs_v)
        for(const auto& name : sm->overlappingIds())
            if (surfs.contains(name))
                overlaps[sm].insert(surfs[name]);

    std::cout << "total surface count (after defective filter): " << surfs.size() << std::endl;
    std::cout << "seed " << seed << " name " << surface_name(seed) << " seed overlapping: "
              << surface_overlaps(&overlaps, seed).size() << "/" << seed->overlappingIds().size() << std::endl;

    SurfacePatchIndex surface_patch_index;
    SurfacePatchIndex* surface_patch_index_ptr = external_surface_patch_index;
    if (use_patch_cache && !surface_patch_index_ptr) {
        std::vector<SurfacePatchIndex::SurfacePtr> patch_surfaces;
        std::set<QuadSurface*> indexed_surfaces;
        patch_surfaces.reserve(surfs.size() + 1);
        auto append_index_surface = [&](QuadSurface * sm) {
            if (!sm)
                return;
            QuadSurface* surf = sm ;
            if (!surf || !indexed_surfaces.insert(surf).second)
                return;
            patch_surfaces.emplace_back(SurfacePatchIndex::SurfacePtr(surf, [](QuadSurface*) {}));
        };
        for (const auto& [_, sm] : surfs) {
            append_index_surface(sm);
        }
        append_index_surface(seed);
        surface_patch_index.setReadOnly(true);
        bool cache_hit = false;
        std::filesystem::path cache_path;
        std::string cache_key;
        cache_key = SurfacePatchIndex::cacheKeyForSurfaces(
            patch_surfaces, surface_patch_index.samplingStride(), 0.0f);
        cache_path = surface_patch_cache_dir / ("surface_patch_index_" + cache_key + ".bin");
        const auto cache_load_start = std::chrono::steady_clock::now();
        cache_hit = surface_patch_index.loadCache(cache_path, patch_surfaces, cache_key);
        const double cache_load_elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - cache_load_start).count();
        std::cout << "SurfacePatchIndex cache "
                  << (cache_hit ? "hit" : "miss")
                  << " key=" << cache_key
                  << " time=" << cache_load_elapsed << "s"
                  << " path=" << cache_path << std::endl;
        if (!cache_hit) {
            const auto rebuild_start = std::chrono::steady_clock::now();
            surface_patch_index.rebuild(patch_surfaces, 0.0f);
            const double rebuild_elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - rebuild_start).count();
            std::cout << "SurfacePatchIndex rebuild time=" << rebuild_elapsed << "s" << std::endl;
            const auto cache_save_start = std::chrono::steady_clock::now();
            const bool saved = surface_patch_index.saveCache(cache_path, cache_key);
            const double cache_save_elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - cache_save_start).count();
            std::cout << "SurfacePatchIndex cache save "
                      << (saved ? "ok" : "failed")
                      << " time=" << cache_save_elapsed << "s"
                      << " path=" << cache_path << std::endl;
        }
        surface_patch_index_ptr = &surface_patch_index;
        std::cout << "SurfacePatchIndex built for " << patch_surfaces.size()
                  << " surfaces patches=" << surface_patch_index.patchCount()
                  << (cache_hit ? " (cache)" : "") << std::endl;
    }

    cv::Mat_<cv::Vec3f> seed_points = seed->rawPoints();
    const bool resume_growth = looks_like_resume_surface(seed, params);
    const int grid_step = std::max(1, static_cast<int>(std::lround(step)));

    int stop_gen = params.value("steps", params.value("grow_steps", 100000));
    stop_gen = std::clamp(stop_gen, 0, 100000);
    int closing_r = 20; //FIXME dont forget to reset!

    // Get sliding window scale from params (set earlier from JSON)

    //1k ~ 1cm, scaled by sliding_w_scale parameter
    int sliding_w = static_cast<int>(1000/src_step/step*2 * sliding_w_scale);
    int w = 2000/src_step/step*2+10+2*closing_r;
    int h = 15000/src_step/step*2+10+2*closing_r;
    bool grow_down = true;
    bool grow_right = true;
    bool grow_up = true;
    bool grow_left = true;
    const bool disable_grid_expansion = params.value("disable_grid_expansion", params.value("fill_growth", false));
    if (params.contains("growth_directions") && params["growth_directions"].is_array()) {
        grow_down = grow_right = grow_up = grow_left = false;
        for (const auto& dir : params["growth_directions"]) {
            if (!dir.is_string()) {
                continue;
            }
            const std::string value = dir.get<std::string>();
            if (value == "all") {
                grow_down = grow_right = grow_up = grow_left = true;
                break;
            }
            if (value == "down") grow_down = true;
            else if (value == "right") grow_right = true;
            else if (value == "up") grow_up = true;
            else if (value == "left") grow_left = true;
        }
        if (!grow_down && !grow_right && !grow_up && !grow_left) {
            grow_down = grow_right = grow_up = grow_left = true;
        }
    }
    const std::vector<cv::Vec2i> neighs = {
        { 1,  0},
        { 1,  1},
        { 0,  1},
        {-1,  1},
        {-1,  0},
        {-1, -1},
        { 0, -1},
        { 1, -1},
    };
    std::cout << "growth directions:"
              << " down=" << grow_down
              << " right=" << grow_right
              << " up=" << grow_up
              << " left=" << grow_left
              << " expand_grid=" << !disable_grid_expansion
              << " steps=" << stop_gen << std::endl;

    const int grid_margin = closing_r + 5;
    const auto requested_extra_lr = [grid_step](const char* key, const nlohmann::json& p) {
        const int extra = std::max(0, p.value(key, 0));
        return (extra + grid_step - 1) / grid_step;
    };
    const int grow_initial_extra_rows_lr = requested_extra_lr("grow_extra_rows", params);
    const int grow_initial_extra_cols_lr = requested_extra_lr("grow_extra_cols", params);
    const int grow_max_extra_rows_lr = std::max(
        grow_initial_extra_rows_lr,
        params.contains("grow_max_extra_rows")
            ? requested_extra_lr("grow_max_extra_rows", params)
            : grow_initial_extra_rows_lr);
    const int grow_max_extra_cols_lr = std::max(
        grow_initial_extra_cols_lr,
        params.contains("grow_max_extra_cols")
            ? requested_extra_lr("grow_max_extra_cols", params)
            : grow_initial_extra_cols_lr);

    int grid_limit_w = std::numeric_limits<int>::max();
    int grid_limit_h = std::numeric_limits<int>::max();
    cv::Point resume_origin(grid_margin, grid_margin);
    cv::Rect resume_grid_bounds;
    int expanded_left = 0;
    int expanded_right = 0;
    int expanded_up = 0;
    int expanded_down = 0;
    int max_extra_left = std::numeric_limits<int>::max();
    int max_extra_right = std::numeric_limits<int>::max();
    int max_extra_up = std::numeric_limits<int>::max();
    int max_extra_down = std::numeric_limits<int>::max();
    if (resume_growth) {
        const int resume_cols = (seed_points.cols + grid_step - 1) / grid_step + 1;
        const int resume_rows = (seed_points.rows + grid_step - 1) / grid_step + 1;
        const int extra_left = grow_left ? grow_initial_extra_cols_lr : 0;
        const int extra_right = grow_right ? grow_initial_extra_cols_lr : 0;
        const int extra_up = grow_up ? grow_initial_extra_rows_lr : 0;
        const int extra_down = grow_down ? grow_initial_extra_rows_lr : 0;
        expanded_left = extra_left;
        expanded_right = extra_right;
        expanded_up = extra_up;
        expanded_down = extra_down;
        max_extra_left = grow_left ? grow_max_extra_cols_lr : 0;
        max_extra_right = grow_right ? grow_max_extra_cols_lr : 0;
        max_extra_up = grow_up ? grow_max_extra_rows_lr : 0;
        max_extra_down = grow_down ? grow_max_extra_rows_lr : 0;
        resume_origin = cv::Point(grid_margin + extra_left, grid_margin + extra_up);
        grid_limit_w = grid_margin + max_extra_left + resume_cols + max_extra_right + grid_margin;
        grid_limit_h = grid_margin + max_extra_up + resume_rows + max_extra_down + grid_margin;
        w = std::max(1, grid_limit_w);
        h = std::max(1, grid_limit_h);
        if (extra_left != max_extra_left || extra_right != max_extra_right) {
            w = std::max(1, grid_margin + extra_left + resume_cols + extra_right + grid_margin);
        }
        if (extra_up != max_extra_up || extra_down != max_extra_down) {
            h = std::max(1, grid_margin + extra_up + resume_rows + extra_down + grid_margin);
        }
        resume_grid_bounds = cv::Rect(resume_origin.x, resume_origin.y, resume_cols, resume_rows);
        std::cout << "resume_growth: true, source grid " << seed_points.size()
                  << " low-res " << cv::Size(resume_cols, resume_rows)
                  << " origin " << resume_origin
                  << " bounds " << cv::Size(w, h)
                  << " initial_extra_lr left=" << extra_left
                  << " right=" << extra_right
                  << " up=" << extra_up
                  << " down=" << extra_down
                  << " max_extra_lr left=" << max_extra_left
                  << " right=" << max_extra_right
                  << " up=" << max_extra_up
                  << " down=" << max_extra_down << std::endl;
    }
    cv::Size size = {w,h};
    cv::Rect bounds(0,0,w-1,h-1);
    cv::Rect save_bounds_inv(closing_r+5,closing_r+5,h-closing_r-10,w-closing_r-10);
    cv::Rect active_bounds(closing_r+5,closing_r+5,w-closing_r-10,h-closing_r-10);
    cv::Rect static_bounds(0,0,0,h);
    const bool constrain_to_resume_grid =
        disable_grid_expansion && resume_growth && resume_grid_bounds.area() > 0;

    int x0 = w/2;
    int y0 = h/2;

    std::cout << "starting with size " << size << " seed " << cv::Vec2i(y0,x0) << std::endl;

    std::unordered_set<cv::Vec2i,vec2i_hash> fringe;

    cv::Mat_<uint8_t> state(size,0);
    cv::Mat_<uint16_t> generations(size, static_cast<uint16_t>(0));
    cv::Mat_<uint16_t> inliers_sum_dbg(size,0);
    cv::Mat_<cv::Vec3d> points(size,{-1,-1,-1});

    cv::Rect used_area(x0,y0,2,2);
    cv::Rect used_area_hr = scaled_rect_trunc(used_area, step);

    SurfTrackerData data;

    bool initialized_from_resume = false;
    if (resume_growth) {
        cv::Mat resume_generations = seed->channel("generations");
        cv::Mat_<uint16_t> resume_generations_u16;
        if (!resume_generations.empty()) {
            if (resume_generations.type() == CV_16UC1) {
                resume_generations_u16 = resume_generations;
            } else {
                resume_generations.convertTo(resume_generations_u16, CV_16U);
            }
        }

        cv::Rect resume_used;
        bool have_resume_used = false;
        cv::Vec2i first_valid(-1, -1);
        int resumed_count = 0;

        for (int ry = 0; ry < seed_points.rows - 1; ry += grid_step) {
            for (int rx = 0; rx < seed_points.cols - 1; rx += grid_step) {
                const cv::Vec2d seed_loc_d{static_cast<double>(ry), static_cast<double>(rx)};
                if (!loc_valid(seed_points, seed_loc_d)) {
                    continue;
                }
                cv::Vec2i p(resume_origin.y + ry / grid_step,
                            resume_origin.x + rx / grid_step);
                if (p[0] < 0 || p[0] >= state.rows || p[1] < 0 || p[1] >= state.cols) {
                    continue;
                }

                data.loc(seed, p) = {static_cast<double>(ry), static_cast<double>(rx)};
                data.surfs(p).insert(seed);
                points(p) = data.lookup_int(seed, p);
                state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
                if (!resume_generations_u16.empty() &&
                    ry < resume_generations_u16.rows &&
                    rx < resume_generations_u16.cols) {
                    generations(p) = std::max<uint16_t>(resume_generations_u16(ry, rx), 1);
                } else {
                    generations(p) = 1;
                }

                cv::Rect cell(p[1], p[0], 1, 1);
                resume_used = have_resume_used ? (resume_used | cell) : cell;
                have_resume_used = true;
                if (first_valid[0] == -1) {
                    first_valid = p;
                }
                resumed_count++;
            }
        }

        if (have_resume_used) {
            used_area = resume_used;
            used_area_hr = scaled_rect_trunc(used_area, step);
            data.seed_loc = first_valid;
            data.seed_coord = points(first_valid);
            initialized_from_resume = true;

            std::set<QuadSurface*> resume_candidates = surface_overlaps(&overlaps, seed);
            if (surface_patch_index_ptr || resume_candidates.empty()) {
                for (const auto& [_, sm] : surfs) {
                    if (sm && sm != seed) {
                        resume_candidates.insert(sm);
                    }
                }
            }

            int fringe_count = 0;
            int discovered_overlap_count = 0;
            for (int j = std::max(0, used_area.y - 1); j <= std::min(used_area.br().y + 1, state.rows - 1); ++j) {
                for (int i = std::max(0, used_area.x - 1); i <= std::min(used_area.br().x + 1, state.cols - 1); ++i) {
                    const cv::Vec2i p(j, i);
                    if ((state(p) & STATE_LOC_VALID) == 0) {
                        continue;
                    }
                    bool has_invalid_neighbor = false;
                    for (const auto& n : neighs) {
                        const cv::Vec2i pn = p + n;
                        if (pn[0] < 0 || pn[0] >= state.rows || pn[1] < 0 || pn[1] >= state.cols ||
                            (state(pn) & STATE_LOC_VALID) == 0) {
                            has_invalid_neighbor = true;
                            break;
                        }
                    }
                    if (!has_invalid_neighbor) {
                        continue;
                    }

                    fringe.insert(p);
                    fringe_count++;
                    const std::size_t before = overlaps[seed].size();
                    for (auto* s : resume_candidates) {
                        add_surface_if_close(s, p, points(p), data, surface_patch_index_ptr);
                        if (data.has(s, p)) {
                            overlaps[seed].insert(s);
                            overlaps[s].insert(seed);
                        }
                    }
                    discovered_overlap_count += static_cast<int>(overlaps[seed].size() - before);
                }
            }

            int seeded_overlap_locations = 0;
            int seeded_overlap_refs = 0;
            if (!surface_overlaps(&overlaps, seed).empty()) {
                for (int j = std::max(0, used_area.y); j <= std::min(used_area.br().y, state.rows - 1); ++j) {
                    for (int i = std::max(0, used_area.x); i <= std::min(used_area.br().x, state.cols - 1); ++i) {
                        const cv::Vec2i p(j, i);
                        if ((state(p) & STATE_LOC_VALID) == 0) {
                            continue;
                        }

                        const std::size_t before = data.surfs(p).size();
                        for (auto* s : surface_overlaps(&overlaps, seed)) {
                            add_surface_if_close(s, p, points(p), data, surface_patch_index_ptr);
                        }
                        const std::size_t added = data.surfs(p).size() - before;
                        if (added > 0) {
                            seeded_overlap_locations++;
                            seeded_overlap_refs += static_cast<int>(added);
                        }
                    }
                }
            }

            std::cout << "resume_growth initialized " << resumed_count
                      << " low-res points, fringe " << fringe_count
                      << ", overlaps " << surface_overlaps(&overlaps, seed).size()
                      << " (discovered " << discovered_overlap_count << ")"
                      << ", seeded overlap refs " << seeded_overlap_refs
                      << " at " << seeded_overlap_locations << " points"
                      << " used_area " << used_area << std::endl;
        } else {
            std::cout << "resume_growth requested but no valid resume samples were found; falling back to center seed" << std::endl;
        }
    }

    if (!initialized_from_resume) {
        cv::Vec2i seed_loc = {seed_points.rows/2, seed_points.cols/2};

        int tries = 0;
        while (seed_points(seed_loc)[0] == -1 || (enforce_z_range && (seed_points(seed_loc)[2] < z_min || seed_points(seed_loc)[2] > z_max))) {
            seed_loc = {rand() % seed_points.rows, rand() % seed_points.cols };
            std::cout << "try loc " << seed_loc << std::endl;
            if (++tries > 10000)
                break;
        }

        data.loc(seed,{y0,x0}) = {
            static_cast<double>(seed_loc[0]),
            static_cast<double>(seed_loc[1])
        };
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

        //insert initial surfs per location
        for(const auto& p : fringe) {
            data.surfs(p).insert(seed);
            cv::Vec3f coord = points(p);
            std::cout << "testing " << p << " from cands: " << surface_overlaps(&overlaps, seed).size() << coord << std::endl;
            for(auto s : surface_overlaps(&overlaps, seed)) {
                auto ptr = s->pointer();
                if (s->pointTo(ptr, coord, same_surface_th, 1000, surface_patch_index_ptr) <= same_surface_th) {
                    cv::Vec3f loc = s->loc_raw(ptr);
                    data.surfs(p).insert(s);
                    data.loc(s, p) = {loc[1], loc[0]};
                }
            }
            std::cout << "fringe point " << p << " surfcount " << data.surfs(p).size() << " init " << data.loc(seed, p) << data.lookup_int(seed, p) << std::endl;
        }
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

    auto count_loc_valid_in_rect = [](const cv::Mat_<uint8_t>& state_, const cv::Rect& area) {
        const cv::Rect safe = area & cv::Rect(0, 0, state_.cols, state_.rows);
        int count = 0;
        for (int y = safe.y; y < safe.br().y; ++y)
            for (int x = safe.x; x < safe.br().x; ++x)
                if (state_(y, x) & STATE_LOC_VALID)
                    ++count;
        return count;
    };

    int best_loc_valid_count = count_loc_valid_in_rect(state, used_area);
    cv::Mat_<cv::Vec3d> best_points = points.clone();
    cv::Mat_<uint8_t> best_state = state.clone();
    cv::Mat_<uint16_t> best_generations = generations.clone();
    SurfTrackerData best_data = data;
    cv::Rect best_used_area = used_area;
    int best_expanded_left = expanded_left;
    int best_expanded_up = expanded_up;

    auto save_best_surface = [&](int count) {
        best_loc_valid_count = count;
        best_points = points.clone();
        best_state = state.clone();
        best_generations = generations.clone();
        best_data = data;
        best_used_area = used_area;
        best_expanded_left = expanded_left;
        best_expanded_up = expanded_up;
    };

    std::vector<SurfTrackerData> data_ths(omp_get_max_threads());
    std::vector<std::vector<cv::Vec2i>> added_points_threads(omp_get_max_threads());
    for(int i=0;i<omp_get_max_threads();i++)
        data_ths[i] = data;

    bool flip_x_done = !flip_x || initialized_from_resume;
    if (flip_x && initialized_from_resume) {
        std::cout << "flip_x skipped for resumed growth" << std::endl;
    }
    bool flip_x_wait_logged = false;

    auto has_horizontal_extent_to_flip = [&]() {
        for (int j = std::max(0, used_area.y); j <= std::min(used_area.br().y + 1, state.rows - 1); ++j) {
            for (int i = std::max(0, used_area.x); i <= std::min(used_area.br().x + 1, state.cols - 1); ++i) {
                if ((state(j, i) & STATE_LOC_VALID) && i != x0) {
                    return true;
                }
            }
        }
        return false;
    };

    auto violates_single_wrap = [&](const cv::Vec2i& p, const cv::Vec3d& coord) {
        if (!single_wrap_umbilicus) {
            return false;
        }

        auto valid_existing_point = [&](const cv::Vec2i& pn) {
            return pn[0] >= 0 && pn[0] < points.rows &&
                   pn[1] >= 0 && pn[1] < points.cols &&
                   (state(pn) & STATE_LOC_VALID) != 0 &&
                   points(pn)[0] != -1;
        };

        auto crosses_existing_point = [&](const cv::Vec2i& pn) {
            return valid_existing_point(pn) &&
                   crosses_single_wrap_seam(*single_wrap_umbilicus, points(pn), coord);
        };

        for (const auto& n : neighs) {
            const cv::Vec2i pn = p + n;
            if (crosses_existing_point(pn)) {
                return true;
            }
        }

        const int max_gap = single_wrap_gap_search > 0
            ? single_wrap_gap_search
            : std::max(points.rows, points.cols);
        const cv::Rect scan_area = (used_area | cv::Rect(p[1], p[0], 1, 1)) &
                                   cv::Rect(0, 0, points.cols, points.rows);
        for (const auto& n : neighs) {
            cv::Vec2i pn = p + n;
            for (int gap = 1; gap <= max_gap; ++gap, pn += n) {
                if (!scan_area.contains(cv::Point(pn[1], pn[0]))) {
                    break;
                }
                if (!valid_existing_point(pn)) {
                    continue;
                }
                if (same_single_wrap_z(points(pn), coord) &&
                    crosses_single_wrap_seam(*single_wrap_umbilicus, points(pn), coord)) {
                    return true;
                }
                break;
            }
        }

        if (single_wrap_ray_angle_deg > 0.0) {
            const cv::Vec3f coord_f = vec3d_to_vec3f(coord);
            const double coord_theta = single_wrap_umbilicus->theta(coord_f);
            const double coord_radius = single_wrap_umbilicus->distance_to_umbilicus(coord_f);
            const int z_index = std::clamp(static_cast<int>(std::lround(coord[2])),
                                           0,
                                           single_wrap_umbilicus->volume_shape()[0] - 1);
            const cv::Vec3f center_f = single_wrap_umbilicus->center_at(z_index);
            const cv::Point2d ray_origin(center_f[0], center_f[1]);
            const cv::Point2d ray_target(coord[0], coord[1]);
            const cv::Point2d ray_dir = ray_target - ray_origin;
            const double ray_len = std::hypot(ray_dir.x, ray_dir.y);

            auto point_at_z_on_edge = [&](const cv::Vec3d& a,
                                          const cv::Vec3d& b,
                                          cv::Point2d& out) {
                const double da = a[2] - coord[2];
                const double db = b[2] - coord[2];
                constexpr double z_eps = 1e-9;
                if (std::abs(da) <= z_eps && std::abs(db) <= z_eps) {
                    out = {a[0], a[1]};
                    return true;
                }
                if ((da < 0.0 && db < 0.0) || (da > 0.0 && db > 0.0) || std::abs(a[2] - b[2]) <= z_eps) {
                    return false;
                }
                const double t = (coord[2] - a[2]) / (b[2] - a[2]);
                if (t < -z_eps || t > 1.0 + z_eps) {
                    return false;
                }
                out = {a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])};
                return true;
            };

            auto radial_segment_conflicts = [&](const cv::Point2d& a, const cv::Point2d& b) {
                if (ray_len <= 0.0) {
                    return false;
                }

                const cv::Point2d seg = b - a;
                const cv::Point2d rel = a - ray_origin;
                const double denom = ray_dir.x * seg.y - ray_dir.y * seg.x;
                const double scale = 1.0 / ray_len;
                if (std::abs(denom) <= 1e-9 * ray_len * std::max(1.0, std::hypot(seg.x, seg.y))) {
                    for (const cv::Point2d& endpoint : {a, b}) {
                        const cv::Point2d d = endpoint - ray_origin;
                        const double radius = std::hypot(d.x, d.y);
                        if (radius <= 0.0) {
                            continue;
                        }
                        const double cross = std::abs(ray_dir.x * d.y - ray_dir.y * d.x) * scale;
                        if (cross <= single_wrap_ray_radius_tolerance &&
                            ray_dir.x * d.x + ray_dir.y * d.y >= 0.0 &&
                            std::abs(radius - coord_radius) > single_wrap_ray_radius_tolerance) {
                            return true;
                        }
                    }
                    return false;
                }

                const double t = (rel.x * seg.y - rel.y * seg.x) / denom;
                const double u = (rel.x * ray_dir.y - rel.y * ray_dir.x) / denom;
                if (t < 0.0 || u < 0.0 || u > 1.0) {
                    return false;
                }

                const double radius = t * ray_len;
                return std::abs(radius - coord_radius) > single_wrap_ray_radius_tolerance;
            };

            auto triangle_conflicts = [&](const cv::Vec3d& a, const cv::Vec3d& b, const cv::Vec3d& c) {
                std::vector<cv::Point2d> intersections;
                intersections.reserve(3);
                cv::Point2d hit;
                if (point_at_z_on_edge(a, b, hit)) {
                    intersections.push_back(hit);
                }
                if (point_at_z_on_edge(b, c, hit)) {
                    intersections.push_back(hit);
                }
                if (point_at_z_on_edge(c, a, hit)) {
                    intersections.push_back(hit);
                }
                if (intersections.size() < 2) {
                    return false;
                }

                return radial_segment_conflicts(intersections[0], intersections[1]);
            };

            const cv::Rect scan_area = used_area & cv::Rect(0, 0, points.cols, points.rows);
            for (int y = scan_area.y; y < scan_area.br().y - 1; ++y) {
                for (int x = scan_area.x; x < scan_area.br().x - 1; ++x) {
                    const cv::Vec2i p00(y, x);
                    const cv::Vec2i p10(y + 1, x);
                    const cv::Vec2i p01(y, x + 1);
                    const cv::Vec2i p11(y + 1, x + 1);
                    if (valid_existing_point(p00) && valid_existing_point(p10) && valid_existing_point(p11) &&
                        triangle_conflicts(points(p00), points(p10), points(p11))) {
                        return true;
                    }
                    if (valid_existing_point(p00) && valid_existing_point(p11) && valid_existing_point(p01) &&
                        triangle_conflicts(points(p00), points(p11), points(p01))) {
                        return true;
                    }
                }
            }

            for (int y = scan_area.y; y < scan_area.br().y; ++y) {
                for (int x = scan_area.x; x < scan_area.br().x; ++x) {
                    const cv::Vec2i pn(y, x);
                    if (pn == p || !valid_existing_point(pn)) {
                        continue;
                    }

                    const cv::Vec3d& existing = points(pn);
                    if (std::abs(existing[2] - coord[2]) > single_wrap_ray_z_tolerance) {
                        continue;
                    }

                    const cv::Vec3f existing_f = vec3d_to_vec3f(existing);
                    if (angular_distance_degrees(single_wrap_umbilicus->theta(existing_f), coord_theta) >
                        single_wrap_ray_angle_deg) {
                        continue;
                    }

                    const double existing_radius = single_wrap_umbilicus->distance_to_umbilicus(existing_f);
                    if (std::abs(existing_radius - coord_radius) > single_wrap_ray_radius_tolerance) {
                        return true;
                    }
                }
            }
        }

        return false;
    };

    auto apply_flip_x = [&]() {
        data.flip_x(x0);

        for(int i=0;i<omp_get_max_threads();i++) {
            data_ths[i] = data;
            added_points_threads[i].clear();
        }

        cv::Mat_<uint8_t> state_orig = state.clone();
        cv::Mat_<uint16_t> generations_orig = generations.clone();
        cv::Mat_<cv::Vec3d> points_orig = points.clone();
        cv::Mat_<uint16_t> inliers_sum_dbg_orig = inliers_sum_dbg.clone();
        state.setTo(0);
        generations.setTo(0);
        points.setTo(cv::Vec3d(-1,-1,-1));
        inliers_sum_dbg.setTo(0);

        cv::Rect new_used_area;
        bool have_used_area = false;
        int clipped = 0;
        for(int j=std::max(0, used_area.y); j<=std::min(used_area.br().y+1, state_orig.rows-1); j++) {
            for(int i=std::max(0, used_area.x); i<=std::min(used_area.br().x+1, state_orig.cols-1); i++) {
                if (state_orig(j, i)) {
                    int nx = x0+x0-i;
                    int ny = j;
                    if (nx < 0 || nx >= state.cols || ny < 0 || ny >= state.rows) {
                        clipped++;
                        continue;
                    }
                    state(ny, nx) = state_orig(j, i);
                    generations(ny, nx) = generations_orig(j, i);
                    points(ny, nx) = points_orig(j, i);
                    inliers_sum_dbg(ny, nx) = inliers_sum_dbg_orig(j, i);
                    cv::Rect cell(nx, ny, 1, 1);
                    new_used_area = have_used_area ? (new_used_area | cell) : cell;
                    have_used_area = true;
                }
            }
        }

        if (clipped > 0) {
            std::cout << "flip_x mirror clipped " << clipped << " points outside the tracing grid" << std::endl;
        }
        if (have_used_area) {
            used_area = new_used_area;
            used_area_hr = scaled_rect_trunc(used_area, step);
        }

        fringe.clear();
        for(int j=std::max(0, used_area.y-2); j<=std::min(used_area.br().y+2, state.rows-1); j++)
            for(int i=std::max(0, used_area.x-2); i<=std::min(used_area.br().x+2, state.cols-1); i++)
                if (state(j,i) & STATE_LOC_VALID)
                    fringe.insert(cv::Vec2i(j,i));

        flip_x_done = true;
        std::cout << "flip_x mirror applied at used_area " << used_area << std::endl;
    };

    bool at_right_border = false;
    std::optional<std::filesystem::path> current_snapshot_path;
    for(int generation=0;generation<stop_gen;generation++) {
        std::unordered_set<cv::Vec2i,vec2i_hash> cands;
        if (generation == 0 && !initialized_from_resume) {
            cands.insert(cv::Vec2i(y0-1,x0));
        }
        else
            for(const auto& p : fringe)
            {
                if ((state(p) & STATE_LOC_VALID) == 0)
                    continue;

                for(const auto& n : neighs) {
                    cv::Vec2i pn = p+n;
                    if (constrain_to_resume_grid &&
                        !resume_grid_bounds.contains(cv::Point(pn[1], pn[0]))) {
                        continue;
                    }
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

        OmpThreadPointCol threadcol(3, cands);

        std::shared_mutex mutex;
        int best_inliers_gen = 0;
#pragma omp parallel
        while (true)
        {
            int r = 1;
            cv::Vec2i p = threadcol.next();

            if (p[0] == -1)
                break;

            if (state(p) & STATE_LOC_VALID)
                continue;

            if (points(p)[0] != -1)
                throw std::runtime_error("oops points(p)[0]");

            std::set<QuadSurface *> local_surfs = {seed};

            mutex.lock_shared();
            SurfTrackerData &data_th = data_ths[omp_get_thread_num()];
            int misses = 0;
            for(const auto& added : added_points_threads[omp_get_thread_num()]) {
                data_th.surfs(added) = data.surfs(added);
                for (auto &s : data.surfsC(added)) {
                    if (!data.has(s, added))
                        std::cout << "where the heck is our data?" << std::endl;
                    else
                        data_th.loc(s, added) = data.loc(s, added);
                }
            }
            mutex.unlock();
            mutex.lock();
            added_points_threads[omp_get_thread_num()].resize(0);
            mutex.unlock();

            for(int oy=std::max(p[0]-r,0);oy<=std::min(p[0]+r,h-1);oy++)
                for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,w-1);ox++)
                    if (state(oy,ox) & STATE_LOC_VALID) {
                        auto p_surfs = data_th.surfsC({oy,ox});
                        local_surfs.insert(p_surfs.begin(), p_surfs.end());
                    }

            cv::Vec3d best_coord = {-1,-1,-1};
            int best_inliers = -1;
            QuadSurface *best_surf = nullptr;
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

                data_th.loc(ref_surf,p) = avg + cv::Vec2d((rand() % 1000)/500.0-1, (rand() % 1000)/500.0-1);

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

                if (!local_metric_consistent(p, coord, state, points, step, src_step)) {
                    data_th.erase(ref_surf, p);
                    continue;
                }

                int inliers_sum = 0;
                int inliers_count = 0;

                //TODO could also have priorities!
                if (approved_sm.contains(ref_surf) && straight_count_init >= 2 && count_init >= 4) {
                    // Respect z-range if enforced
                    if (enforce_z_range && (coord[2] < z_min || coord[2] > z_max)) {
                        data_th.erase(ref_surf, p);
                        continue;
                    }
                    std::cout << "found approved sm " << ref_surf->id << std::endl;

                    // Log approved surface if not already logged
                    if (used_approved_names.insert(ref_surf->id).second) {
                        mutex.lock();
                        approved_log << ref_surf->id << std::endl;
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

                for(auto test_surf : local_surfs) {
                    auto ptr = test_surf->pointer();
                    //FIXME this does not check geometry, only if its also on the surfaces (which might be good enough...)
                    if (test_surf->pointTo(ptr, coord, same_surface_th, 10,
                                                       surface_patch_index_ptr) <= same_surface_th) {
                        int count = 0;
                        int straight_count = 0;
                        state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
                        cv::Vec3f loc = test_surf->loc_raw(ptr);
                        data_th.loc(test_surf, p) = {loc[1], loc[0]};
                        float cost = local_cost(test_surf, p, data_th, state, points, step, src_step, &count, &straight_count);
                        state(p) = 0;
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

            if (!best_approved && (best_inliers >= curr_best_inl_th || best_ref_seed))
            {
                if (enforce_z_range && (best_coord[2] < z_min || best_coord[2] > z_max)) {
                    // Final guard: reject best candidate outside z-range
                    best_inliers = -1;
                    best_ref_seed = false;
                } else if (!local_metric_consistent(p, best_coord, state, points, step, src_step)) {
                    best_inliers = -1;
                    best_ref_seed = false;
                } else {
                cv::Vec2f tmp_loc_;
                cv::Rect used_th = used_area;
                if (used_th.width > 3 && used_th.height > 3) {
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
            }

            if (best_inliers >= curr_best_inl_th || best_ref_seed) {
                if (best_coord[0] == -1)
                    throw std::runtime_error("oops best_cord[0]");
                if (violates_single_wrap(p, best_coord)) {
                    state(p) = 0;
                    generations(p) = 0;
                    points(p) = {-1,-1,-1};
                    continue;
                }

                data_th.surfs(p).insert(best_surf);
                data_th.loc(best_surf, p) = best_loc;
                state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
                generations(p) = static_cast<uint16_t>(std::min(generation + 1, static_cast<int>(std::numeric_limits<uint16_t>::max())));
                points(p) = best_coord;
                inliers_sum_dbg(p) = best_inliers;

                ceres::Problem problem;
                surftrack_add_local(best_surf, p, data_th, problem, state, points, step, src_step, SURF_LOSS | LOSS_ZLOC);

                std::set<QuadSurface *> more_local_surfs;

                for(auto test_surf : local_surfs) {
                    for(auto s : surface_overlaps(&overlaps, test_surf))
                        if (!local_surfs.contains(s) && s != best_surf)
                            more_local_surfs.insert(s);

                    if (test_surf == best_surf)
                        continue;

                    auto ptr = test_surf->pointer();
                    if (test_surf->pointTo(ptr, best_coord, same_surface_th, 10,
                                                       surface_patch_index_ptr) <= same_surface_th) {
                        cv::Vec3f loc = test_surf->loc_raw(ptr);
                        data_th.loc(test_surf, p) = {loc[1], loc[0]};
                        int count = 0;
                        float cost = local_cost(test_surf, p, data_th, state, points, step, src_step, &count);
                        //FIXME opt then check all in extra again!
                        if (cost < local_cost_inl_th) {
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
                for(auto test_surf : more_local_surfs) {
                    auto ptr = test_surf->pointer();
                    float res = test_surf->pointTo(ptr, best_coord, same_surface_th, 10,
                                                               surface_patch_index_ptr);
                    if (res <= same_surface_th) {
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

                data.surfs(p) = data_th.surfs(p);
                for(auto &s : data.surfs(p))
                    if (data_th.has(s, p))
                        data.loc(s, p) = data_th.loc(s, p);

                for(int t=0;t<omp_get_max_threads();t++)
                    added_points_threads[t].push_back(p);

                if (!used_area.contains(cv::Point(p[1],p[0]))) {
                    used_area = used_area | cv::Rect(p[1],p[0],1,1);
                    used_area_hr = scaled_rect_trunc(used_area, step);
                }
                fringe.insert(p);
                mutex.unlock();
            }
            else if (best_inliers == -1) {
                //just try again some other time
                state(p) = 0;
                generations(p) = 0;
                points(p) = {-1,-1,-1};
            }
            else {
                state(p) = 0;
                generations(p) = 0;
                points(p) = {-1,-1,-1};
#pragma omp critical
                best_inliers_gen = std::max(best_inliers_gen, best_inliers);
            }
        }

        if (!flip_x_done && generation >= 1) {
            if (has_horizontal_extent_to_flip()) {
                apply_flip_x();
            } else if (!flip_x_wait_logged) {
                std::cout << "flip_x waiting for non-center X extent before mirroring" << std::endl;
                flip_x_wait_logged = true;
            }
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
                if (active.area() > 0) {
                    for(int j=std::max(0, active.y-2);j<=std::min(active.br().y+2, state.rows-1);j++)
                        for(int i=std::max(0, active.x-2);i<=std::min(active.br().x+2, state.cols-1);i++)
                            if (state(j,i) & STATE_LOC_VALID)
                                    fringe.insert(cv::Vec2i(j,i));
                }
            }
        }
        else
            curr_best_inl_th = inlier_base_threshold;

        loc_valid_count = 0;
        for(int j=used_area.y;j<used_area.br().y-1;j++)
            for(int i=used_area.x;i<used_area.br().x-1;i++)
                if (state(j,i) & STATE_LOC_VALID)
                    loc_valid_count++;
        if (loc_valid_count > best_loc_valid_count) {
            save_best_surface(loc_valid_count);
        }

        bool update_mapping = (succ >= 1000 && (loc_valid_count-last_succ_parametrization) >= std::max(100.0, 0.3*last_succ_parametrization));
        if (fringe.empty() && final_opts) {
            final_opts--;
            update_mapping = true;
        }

        if (!global_steps_per_window)
            update_mapping = false;

        if (generation % 50 == 0 || update_mapping /*|| generation < 10*/) {
            {
                cv::Mat_<cv::Vec3d> points_hr = surftrack_genpoints_hr(data, state, points, used_area, step, src_step);
                cv::Mat_<uint16_t> generations_hr = surftrack_generations_hr(state, generations, used_area, step);
                auto dbg_surf = new QuadSurface(points_hr(used_area_hr), {1/src_step,1/src_step});
                dbg_surf->setChannel("generations", generations_hr(used_area_hr));
                dbg_surf->meta = utils::Json::object();
                dbg_surf->meta["vc_grow_seg_from_segments_params"] = json_to_utils(params);

                float const area_est_vx2 = loc_valid_count*src_step*src_step*step*step;
                float const area_est_cm2 = area_est_vx2 * voxelsize * voxelsize / 1e8;
                dbg_surf->meta["area_vx2"] = static_cast<double>(area_est_vx2);
                dbg_surf->meta["area_cm2"] = static_cast<double>(area_est_cm2);
                dbg_surf->meta["used_approved_segments"] = json_to_utils(nlohmann::json(std::vector<std::string>(used_approved_names.begin(), used_approved_names.end())));
                const std::filesystem::path legacy_current_path = tgt_dir / (std::string(Z_DBG_GEN_PREFIX)+"current");
                std::string uuid = std::string(Z_DBG_GEN_PREFIX) + get_surface_time_str();
                const std::filesystem::path snapshot_path = tgt_dir / uuid;
                dbg_surf->save(snapshot_path.string(), uuid, true);

                std::error_code cleanup_ec;
                if (legacy_current_path != snapshot_path) {
                    std::filesystem::remove_all(legacy_current_path, cleanup_ec);
                }
                cleanup_ec.clear();
                if (current_snapshot_path && *current_snapshot_path != snapshot_path) {
                    std::filesystem::remove_all(*current_snapshot_path, cleanup_ec);
                }
                current_snapshot_path = snapshot_path;
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

            cv::Rect active = active_bounds & used_area;
            if (active.area() > 0) {
                auto count_loc_valid_in = [](const cv::Mat_<uint8_t>& state_, const cv::Rect& area) {
                    int count = 0;
                    for (int y = area.y; y < area.br().y; ++y)
                        for (int x = area.x; x < area.br().x; ++x)
                            if (state_(y, x) & STATE_LOC_VALID)
                                ++count;
                    return count;
                };
                const int valid_before_opt = count_loc_valid_in(state, active);

                optimize_surface_mapping(opt_data, opt_state, opt_points, active, static_bounds, step, src_step,
                                         opt_data.seed_loc, closing_r, true, tgt_dir, surface_patch_index_ptr, &overlaps, debug_images);
                const int valid_after_opt = count_loc_valid_in(opt_state, active);
                if (valid_before_opt > 0 && valid_after_opt * 2 < valid_before_opt) {
                    std::cout << "optimizer: rejecting mapping; valid points collapsed "
                              << valid_before_opt << " -> " << valid_after_opt << std::endl;
                } else {
                    copy(opt_data, data, active);
                    opt_points(active).copyTo(points(active));
                    opt_state(active).copyTo(state(active));
                }

                for(int i=0;i<omp_get_max_threads();i++) {
                    data_ths[i] = data;
                    added_points_threads[i].resize(0);
                }
            }

            last_succ_parametrization = loc_valid_count;
            //recalc fringe after surface optimization (which often shrinks the surf)
            fringe.clear();
            curr_best_inl_th = inlier_base_threshold;
            if (active.area() > 0) {
                for(int j=std::max(0, active.y-2);j<=std::min(active.br().y+2, state.rows-1);j++)
                    for(int i=std::max(0, active.x-2);i<=std::min(active.br().x+2, state.cols-1);i++)
                        if (state(j,i) & STATE_LOC_VALID)
                            fringe.insert(cv::Vec2i(j,i));
            }

            if (debug_images) {
                cv::Mat_<cv::Vec3d> points_hr = surftrack_genpoints_hr(data, state, points, used_area, step, src_step);
                cv::Mat_<uint16_t> generations_hr = surftrack_generations_hr(state, generations, used_area, step);
                auto dbg_surf = new QuadSurface(points_hr(used_area_hr), {1/src_step,1/src_step});
                dbg_surf->setChannel("generations", generations_hr(used_area_hr));
                dbg_surf->meta = utils::Json::object();
                dbg_surf->meta["vc_grow_seg_from_segments_params"] = json_to_utils(params);

                std::string uuid = std::string(Z_DBG_GEN_PREFIX)+"opt";
                float const area_est_vx2 = loc_valid_count*src_step*src_step*step*step;
                float const area_est_cm2 = area_est_vx2 * voxelsize * voxelsize / 1e8;
                dbg_surf->meta["area_vx2"] = static_cast<double>(area_est_vx2);
                dbg_surf->meta["area_cm2"] = static_cast<double>(area_est_cm2);
                dbg_surf->meta["used_approved_segments"] = json_to_utils(nlohmann::json(std::vector<std::string>(used_approved_names.begin(), used_approved_names.end())));
                dbg_surf->save(tgt_dir / uuid, uuid, true);
                delete dbg_surf;
            }
        }

        float const current_area_vx2 = loc_valid_count*src_step*src_step*step*step;
        float const current_area_cm2 = current_area_vx2 * voxelsize * voxelsize / 1e8;
        printf("gen %d processing %lu fringe cands (total done %d fringe: %lu) area %.0f vx^2 (%f cm^2) best th: %d\n",
               generation, static_cast<unsigned long>(cands.size()), succ, static_cast<unsigned long>(fringe.size()),
               current_area_vx2, current_area_cm2, best_inliers_gen);

        //continue expansion
        const int max_grid_extent = std::max(1, static_cast<int>(max_width / step));
        const int max_grid_w = std::min(max_grid_extent, grid_limit_w);
        const int max_grid_h = std::min(max_grid_extent, grid_limit_h);
        const auto remaining_extra = [](int max_extra, int expanded) {
            if (max_extra == std::numeric_limits<int>::max()) {
                return std::numeric_limits<int>::max();
            }
            return std::max(0, max_extra - expanded);
        };
        const int remaining_left = remaining_extra(max_extra_left, expanded_left);
        const int remaining_right = remaining_extra(max_extra_right, expanded_right);
        const int remaining_up = remaining_extra(max_extra_up, expanded_up);
        const int remaining_down = remaining_extra(max_extra_down, expanded_down);
        const bool can_expand_left = !disable_grid_expansion && grow_left && remaining_left > 0 && w < max_grid_w;
        const bool can_expand_right = !disable_grid_expansion && grow_right && remaining_right > 0 && w < max_grid_w;
        const bool can_expand_up = !disable_grid_expansion && grow_up && remaining_up > 0 && h < max_grid_h;
        const bool can_expand_down = !disable_grid_expansion && grow_down && remaining_down > 0 && h < max_grid_h;
        if (fringe.empty() && (can_expand_left || can_expand_right || can_expand_up || can_expand_down))
        {
            at_right_border = false;
            int width_capacity = max_grid_w - w;
            int height_capacity = max_grid_h - h;
            const int add_left = can_expand_left ? std::min({sliding_w, remaining_left, width_capacity}) : 0;
            width_capacity -= add_left;
            const int add_right = can_expand_right ? std::min({sliding_w, remaining_right, width_capacity}) : 0;
            const int add_up = can_expand_up ? std::min({sliding_w, remaining_up, height_capacity}) : 0;
            height_capacity -= add_up;
            const int add_down = can_expand_down ? std::min({sliding_w, remaining_down, height_capacity}) : 0;
            const int new_w = w + add_left + add_right;
            const int new_h = h + add_up + add_down;
            if (new_w == w && new_h == h) {
                break;
            }
            std::cout << "expanding by left=" << add_left
                      << " right=" << add_right
                      << " up=" << add_up
                      << " down=" << add_down << std::endl;

            std::cout << size << bounds << save_bounds_inv << used_area << active_bounds << (used_area & active_bounds) << static_bounds << std::endl;
            final_opts = global_steps_per_window;
            w = new_w;
            h = new_h;
            expanded_left += add_left;
            expanded_right += add_right;
            expanded_up += add_up;
            expanded_down += add_down;
            size = {w,h};
            bounds = {0,0,w-1,h-1};
            save_bounds_inv = {closing_r+5,closing_r+5,h-closing_r-10,w-closing_r-10};

            const cv::Vec2i grid_offset(add_up, add_left);
            cv::Mat_<cv::Vec3d> old_points = points;
            const cv::Rect copy_dst(add_left, add_up, old_points.cols, old_points.rows);
            points = cv::Mat_<cv::Vec3d>(size, {-1,-1,-1});
            old_points.copyTo(points(copy_dst));

            cv::Mat_<uint8_t> old_state = state;
            state = cv::Mat_<uint8_t>(size, 0);
            old_state.copyTo(state(copy_dst));

            cv::Mat_<uint16_t> old_generations = generations;
            generations = cv::Mat_<uint16_t>(size, static_cast<uint16_t>(0));
            old_generations.copyTo(generations(copy_dst));

            cv::Mat_<uint16_t> old_inliers_sum_dbg = inliers_sum_dbg;
            inliers_sum_dbg = cv::Mat_<uint16_t>(size, 0);
            old_inliers_sum_dbg.copyTo(inliers_sum_dbg(copy_dst));

            if (grid_offset != cv::Vec2i(0, 0)) {
                data.translate(grid_offset);
                used_area += cv::Point(add_left, add_up);
                used_area_hr = scaled_rect_trunc(used_area, step);
                x0 += add_left;
                y0 += add_up;
            }

            int overlap = 5;
            const int inner_x0 = closing_r + 5;
            const int inner_y0 = closing_r + 5;
            const int inner_x1 = w - closing_r - 5;
            const int inner_y1 = h - closing_r - 5;
            const auto active_axis = [overlap](int add_before,
                                               int add_after,
                                               int used_start,
                                               int used_end,
                                               int inner_start,
                                               int inner_end) {
                if (add_before > 0 && add_after > 0) {
                    return std::pair<int, int>(inner_start, inner_end);
                }
                if (add_before > 0) {
                    return std::pair<int, int>(
                        std::max(inner_start, used_start - add_before - overlap),
                        std::min(inner_end, used_start + overlap));
                }
                if (add_after > 0) {
                    return std::pair<int, int>(
                        std::max(inner_start, used_end - overlap),
                        std::min(inner_end, used_end + add_after + overlap));
                }
                return std::pair<int, int>(inner_start, inner_end);
            };
            const auto [active_x0, active_x1] = active_axis(
                add_left, add_right, used_area.x, used_area.br().x, inner_x0, inner_x1);
            const auto [active_y0, active_y1] = active_axis(
                add_up, add_down, used_area.y, used_area.br().y, inner_y0, inner_y1);
            active_bounds = cv::Rect(active_x0,
                                     active_y0,
                                     std::max(0, active_x1 - active_x0),
                                     std::max(0, active_y1 - active_y0));
            static_bounds = cv::Rect(0, 0, w, h);
            if (add_left == 0 && add_right > 0) {
                static_bounds = cv::Rect(0, 0, std::max(0, active_x0), h);
            } else if (add_right == 0 && add_left > 0) {
                static_bounds = cv::Rect(std::min(w, active_x1), 0,
                                         std::max(0, w - active_x1), h);
            } else if (add_up == 0 && add_down > 0) {
                static_bounds = cv::Rect(0, 0, w, std::max(0, active_y0));
            } else if (add_down == 0 && add_up > 0) {
                static_bounds = cv::Rect(0, std::min(h, active_y1),
                                         w, std::max(0, h - active_y1));
            }

            for(int i=0;i<omp_get_max_threads();i++) {
                data_ths[i] = data;
                added_points_threads[i].clear();
            }

            cv::Rect active = active_bounds & used_area;

            std::cout << size << bounds << save_bounds_inv << used_area << active_bounds << (used_area & active_bounds) << static_bounds << std::endl;
            fringe.clear();
            curr_best_inl_th = inlier_base_threshold;
            if (active.area() > 0) {
                for(int j=std::max(0, active.y-2);j<=std::min(active.br().y+2, state.rows-1);j++)
                    for(int i=std::max(0, active.x-2);i<=std::min(active.br().x+2, state.cols-1);i++)
                        if (state(j,i) & STATE_LOC_VALID)
                            fringe.insert(cv::Vec2i(j,i));
            }
        }

        cv::imwrite((tgt_dir / "inliers_sum.tif").string(), inliers_sum_dbg(used_area));

        if (fringe.empty())
            break;
    }

    approved_log.close();

    const int final_loc_valid_count = count_loc_valid_in_rect(state, used_area);
    if (best_loc_valid_count > final_loc_valid_count) {
        std::cout << "using best growth snapshot; final valid count shrank "
                  << final_loc_valid_count << " -> " << best_loc_valid_count << std::endl;
        points = best_points.clone();
        state = best_state.clone();
        generations = best_generations.clone();
        data = best_data;
        used_area = best_used_area;
        used_area_hr = scaled_rect_trunc(used_area, step);
        expanded_left = best_expanded_left;
        expanded_up = best_expanded_up;
        loc_valid_count = best_loc_valid_count;
    } else {
        loc_valid_count = final_loc_valid_count;
    }

    float const area_est_vx2 = loc_valid_count*src_step*src_step*step*step;
    float const area_est_cm2 = area_est_vx2 * voxelsize * voxelsize / 1e8;
    std::cout << "area est: " << area_est_vx2 << " vx^2 (" << area_est_cm2 << " cm^2)" << std::endl;

    cv::Mat_<cv::Vec3d> points_hr = surftrack_genpoints_hr(data, state, points, used_area, step, src_step);
    cv::Mat_<uint16_t> generations_hr = surftrack_generations_hr(state, generations, used_area, step);

    auto surf = new QuadSurface(points_hr(used_area_hr), {1/src_step,1/src_step});
    surf->setChannel("generations", generations_hr(used_area_hr));

    surf->meta = utils::Json::object();
    surf->meta["area_vx2"] = static_cast<double>(area_est_vx2);
    surf->meta["area_cm2"] = static_cast<double>(area_est_cm2);
    surf->meta["used_approved_segments"] = json_to_utils(nlohmann::json(std::vector<std::string>(used_approved_names.begin(), used_approved_names.end())));
    if (resume_growth) {
        const int offset_col = grid_margin + expanded_left - used_area.x;
        const int offset_row = grid_margin + expanded_up - used_area.y;
        auto off_arr = utils::Json::array();
        off_arr.push_back(offset_col);
        off_arr.push_back(offset_row);
        surf->meta["grid_offset"] = std::move(off_arr);
    }

    return surf;
}

QuadSurface *grow_surf_from_surfs(QuadSurface *seed, const std::vector<QuadSurface*> &surfs_v, const utils::Json &params, float voxelsize)
{
    return grow_surf_from_surfs(seed, surfs_v, params, voxelsize, nullptr);
}

QuadSurface *grow_surf_from_surfs(QuadSurface *seed,
                                  const std::vector<QuadSurface*> &surfs_v,
                                  const utils::Json &params,
                                  float voxelsize,
                                  SurfacePatchIndex* surface_patch_index)
{
    if (!seed)
        return nullptr;

    std::vector<QuadSurface*> grow_surfs;
    std::set<QuadSurface*> seen;
    QuadSurface* seed_surf = nullptr;
    for (QuadSurface* surf : surfs_v) {
        if (!surf || !seen.insert(surf).second)
            continue;
        grow_surfs.push_back(surf);
        if (surf == seed)
            seed_surf = surf;
    }

    if (!seed_surf) {
        grow_surfs.push_back(seed);
        seed_surf = seed;
    }

    nlohmann::json grow_params = json_from_utils(params);
    return grow_surf_from_surfs_impl(seed_surf, grow_surfs, grow_params, voxelsize, surface_patch_index);
}
