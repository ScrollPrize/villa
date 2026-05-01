#include <omp.h>


#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/PointIndex.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/tracer/SurfaceModeling.hpp"
#include "vc/core/util/OMPThreadPointCollection.hpp"
#include "vc/core/util/DateTime.hpp"
#include "vc/core/util/HashFunctions.hpp"
#include "vc/core/types/ChunkedTensor.hpp"

#include "growth_strategies/CandidateOrdering.hpp"
#include "growth_strategies/GrowthConfig.hpp"
#include "SurfTrackerData.hpp"

#include "vc/core/util/LifeTime.hpp"
#include "vc/tracer/Tracer.hpp"
#include "utils/Json.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
static float straight_min_count = 1.0f;    // Minimum number of straight constraints
static int inlier_base_threshold = 20;     // Starting threshold for inliers
static int inlier_threshold_drop_step = 2; // Maximum retry drop per stalled generation
static constexpr int rollout_warmup_generations = 30;

static double deterministic_probe_noise(uint32_t y, uint32_t x, uint32_t salt)
{
    uint32_t v = y * 0x9E3779B9u ^ x * 0x85EBCA6Bu ^ salt * 0xC2B2AE35u;
    v ^= v >> 16;
    v *= 0x7FEB352Du;
    v ^= v >> 15;
    v *= 0x846CA68Bu;
    v ^= v >> 16;
    return static_cast<double>(v & 0xFFFFu) / 32768.0 - 1.0;
}

static cv::Vec2d deterministic_probe_jitter(const cv::Vec2i& p)
{
    return {
        deterministic_probe_noise(static_cast<uint32_t>(p[0]), static_cast<uint32_t>(p[1]), 0),
        deterministic_probe_noise(static_cast<uint32_t>(p[0]), static_cast<uint32_t>(p[1]), 1)
    };
}

struct LocalSurfaceLocSample
{
    cv::Vec2d grid_offset;
    cv::Vec2d loc;
};

static cv::Vec2d median_surface_loc(std::vector<LocalSurfaceLocSample> samples)
{
    auto median_component = [&](int component) {
        std::vector<double> values;
        values.reserve(samples.size());
        for (const auto& sample : samples) {
            values.push_back(sample.loc[component]);
        }
        const size_t mid = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
        double median = values[mid];
        if ((values.size() % 2) == 0) {
            std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid - 1), values.end());
            median = 0.5 * (median + values[mid - 1]);
        }
        return median;
    };

    return {median_component(0), median_component(1)};
}

static std::optional<cv::Vec2d> affine_surface_loc_prediction(
    const std::vector<LocalSurfaceLocSample>& samples,
    const std::optional<size_t>& skip_sample = std::nullopt)
{
    const size_t fit_count = samples.size() - (skip_sample.has_value() ? 1 : 0);
    if (fit_count < 3) {
        return std::nullopt;
    }

    cv::Mat_<double> a(static_cast<int>(fit_count), 3);
    cv::Mat_<double> bu(static_cast<int>(fit_count), 1);
    cv::Mat_<double> bv(static_cast<int>(fit_count), 1);
    int row = 0;
    for (size_t idx = 0; idx < samples.size(); ++idx) {
        if (skip_sample.has_value() && idx == *skip_sample) {
            continue;
        }
        const auto& sample = samples[idx];
        a(row, 0) = 1.0;
        a(row, 1) = sample.grid_offset[0];
        a(row, 2) = sample.grid_offset[1];
        bu(row, 0) = sample.loc[0];
        bv(row, 0) = sample.loc[1];
        ++row;
    }

    cv::Mat_<double> xu;
    cv::Mat_<double> xv;
    if (!cv::solve(a, bu, xu, cv::DECOMP_SVD) || !cv::solve(a, bv, xv, cv::DECOMP_SVD)) {
        return std::nullopt;
    }

    return cv::Vec2d(xu(0, 0), xv(0, 0));
}

static cv::Vec2d robust_affine_surface_loc_prediction(const std::vector<LocalSurfaceLocSample>& samples)
{
    if (samples.empty()) {
        return {-1, -1};
    }
    if (samples.size() < 3) {
        return median_surface_loc(samples);
    }

    auto prediction = affine_surface_loc_prediction(samples);
    if (!prediction.has_value()) {
        return median_surface_loc(samples);
    }

    if (samples.size() < 4) {
        return *prediction;
    }

    size_t worst_idx = 0;
    double worst_residual_sq = -1.0;
    for (size_t idx = 0; idx < samples.size(); ++idx) {
        const auto& sample = samples[idx];
        cv::Matx<double, 1, 3> row(1.0, sample.grid_offset[0], sample.grid_offset[1]);

        cv::Mat_<double> a(static_cast<int>(samples.size()), 3);
        cv::Mat_<double> bu(static_cast<int>(samples.size()), 1);
        cv::Mat_<double> bv(static_cast<int>(samples.size()), 1);
        for (size_t fit_idx = 0; fit_idx < samples.size(); ++fit_idx) {
            const auto& fit_sample = samples[fit_idx];
            a(static_cast<int>(fit_idx), 0) = 1.0;
            a(static_cast<int>(fit_idx), 1) = fit_sample.grid_offset[0];
            a(static_cast<int>(fit_idx), 2) = fit_sample.grid_offset[1];
            bu(static_cast<int>(fit_idx), 0) = fit_sample.loc[0];
            bv(static_cast<int>(fit_idx), 0) = fit_sample.loc[1];
        }
        cv::Mat_<double> xu;
        cv::Mat_<double> xv;
        if (!cv::solve(a, bu, xu, cv::DECOMP_SVD) || !cv::solve(a, bv, xv, cv::DECOMP_SVD)) {
            return median_surface_loc(samples);
        }

        const cv::Vec2d fitted(
            row(0, 0) * xu(0, 0) + row(0, 1) * xu(1, 0) + row(0, 2) * xu(2, 0),
            row(0, 0) * xv(0, 0) + row(0, 1) * xv(1, 0) + row(0, 2) * xv(2, 0));
        const cv::Vec2d residual = fitted - sample.loc;
        const double residual_sq = residual.dot(residual);
        if (residual_sq > worst_residual_sq) {
            worst_residual_sq = residual_sq;
            worst_idx = idx;
        }
    }

    auto trimmed_prediction = affine_surface_loc_prediction(samples, worst_idx);
    return trimmed_prediction.value_or(*prediction);
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

static int remove_components_disconnected_from_seed(SurfTrackerData& data,
                                                    cv::Mat_<uint8_t>& state,
                                                    cv::Mat_<cv::Vec3d>& points)
{
    const cv::Vec2i seed = data.seed_loc;
    if (seed[0] < 0 || seed[0] >= state.rows || seed[1] < 0 || seed[1] >= state.cols ||
        !(state(seed) & STATE_LOC_VALID)) {
        std::cout << "connected components: seed is not valid, skipping cleanup at "
                  << seed << std::endl;
        return 0;
    }

    cv::Mat_<uint8_t> reachable(state.size(), static_cast<uint8_t>(0));
    std::vector<cv::Vec2i> queue;
    queue.reserve(static_cast<size_t>(std::max(1, state.rows * state.cols / 8)));
    reachable(seed) = 1;
    queue.push_back(seed);

    const cv::Vec2i offsets[] = {
        {-1, -1},
        {-1,  0},
        {-1,  1},
        { 0, -1},
        { 0,  1},
        { 1, -1},
        { 1,  0},
        { 1,  1},
    };

    for (size_t idx = 0; idx < queue.size(); ++idx) {
        const cv::Vec2i p = queue[idx];
        for (const cv::Vec2i& offset : offsets) {
            const cv::Vec2i q = p + offset;
            if (q[0] < 0 || q[0] >= state.rows || q[1] < 0 || q[1] >= state.cols)
                continue;
            if (reachable(q) || !(state(q) & STATE_LOC_VALID))
                continue;
            reachable(q) = 1;
            queue.push_back(q);
        }
    }

    int removed = 0;
    for (int j = 0; j < state.rows; ++j) {
        for (int i = 0; i < state.cols; ++i) {
            const cv::Vec2i p(j, i);
            if (!(state(p) & STATE_LOC_VALID) || reachable(p))
                continue;

            std::set<QuadSurface*> surf_src = data.surfsC(p);
            for (QuadSurface* surf : surf_src) {
                data.erase(surf, p);
                data.eraseSurf(surf, p);
            }
            data._surfs.erase(p);
            state(p) = 0;
            points(p) = {-1, -1, -1};
            ++removed;
        }
    }

    if (removed > 0) {
        std::cout << "connected components: removed " << removed
                  << " points disconnected from seed " << seed << std::endl;
    }
    return removed;
}

static std::set<QuadSurface*> surface_patch_candidates(SurfacePatchIndex* surface_patch_index,
                                                       const cv::Vec3d& coord,
                                                       QuadSurface* exclude = nullptr)
{
    std::set<QuadSurface*> candidates;
    if (!surface_patch_index || surface_patch_index->empty() || coord[0] == -1) {
        return candidates;
    }

    SurfacePatchIndex::PointQuery query;
    query.worldPoint = {
        static_cast<float>(coord[0]),
        static_cast<float>(coord[1]),
        static_cast<float>(coord[2])
    };
    query.tolerance = same_surface_th;

    for (const auto& surface : surface_patch_index->locateSurfaces(query)) {
        QuadSurface* raw = surface.get();
        if (raw && raw != exclude) {
            candidates.insert(raw);
        }
    }
    return candidates;
}

static std::vector<SurfacePatchIndex::LookupResult> surface_patch_hits(SurfacePatchIndex* surface_patch_index,
                                                                       const cv::Vec3d& coord,
                                                                       QuadSurface* exclude = nullptr)
{
    std::vector<SurfacePatchIndex::LookupResult> hits;
    if (!surface_patch_index || surface_patch_index->empty() || coord[0] == -1) {
        return hits;
    }

    SurfacePatchIndex::PointQuery query;
    query.worldPoint = {
        static_cast<float>(coord[0]),
        static_cast<float>(coord[1]),
        static_cast<float>(coord[2])
    };
    query.tolerance = same_surface_th;

    for (auto hit : surface_patch_index->locateAll(query)) {
        QuadSurface* raw = hit.surface.get();
        if (raw && raw != exclude) {
            hits.push_back(std::move(hit));
        }
    }
    return hits;
}

static int add_surface_patch_candidates(const cv::Vec2i& p,
                                        const cv::Vec3d& coord,
                                        SurfTrackerData& data,
                                        SurfacePatchIndex* surface_patch_index,
                                        QuadSurface* exclude = nullptr)
{
    if (!surface_patch_index || surface_patch_index->empty() || coord[0] == -1) {
        return 0;
    }

    SurfacePatchIndex::PointQuery query;
    query.worldPoint = {
        static_cast<float>(coord[0]),
        static_cast<float>(coord[1]),
        static_cast<float>(coord[2])
    };
    query.tolerance = same_surface_th;

    int added = 0;
    for (const auto& hit : surface_patch_index->locateAll(query)) {
        QuadSurface* surf = hit.surface.get();
        if (!surf || surf == exclude || data.has(surf, p)) {
            continue;
        }
        const cv::Vec3f loc = surf->loc_raw(hit.ptr);
        if (SurfTrackerData::lookup_int_loc(surf, {loc[1], loc[0]})[0] == -1) {
            continue;
        }
        data.surfs(p).insert(surf);
        data.loc(surf, p) = {loc[1], loc[0]};
        ++added;
    }
    return added;
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


static double local_cost(QuadSurface *sm, const cv::Vec2i& p, SurfTrackerData &data, const cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points,
    float step, float src_step, int *ref_count = nullptr, int *straight_count_ptr = nullptr,
    const cv::Vec3d* candidate_coord = nullptr)
{
    int straigh_count;
    if (!straight_count_ptr)
        straight_count_ptr = &straigh_count;

    const cv::Vec3d saved_point = points(p);
    if (candidate_coord) {
        points(p) = *candidate_coord;
    }

    double test_loss = 0.0;
    ceres::Problem problem_test;

    int count = surftrack_add_local(sm, p, data, problem_test, state, points, step, src_step, 0, straight_count_ptr);
    if (ref_count)
        *ref_count = count;

    problem_test.Evaluate(ceres::Problem::EvaluateOptions(), &test_loss, nullptr, nullptr, nullptr);
    if (candidate_coord) {
        points(p) = saved_point;
    }

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
    const int hr_rows = static_cast<int>(state.rows * step);
    const int hr_cols = static_cast<int>(state.cols * step);
    cv::Mat_<cv::Vec3f> points_hr(hr_rows, hr_cols, {0,0,0});
    cv::Mat_<int> counts_hr(hr_rows, hr_cols, 0);
    cv::Mat_<uint8_t> rejected_hr(hr_rows, hr_cols, static_cast<uint8_t>(0));
    const double sample_outlier_dist = std::max<double>(step_src * 3.0, step * step_src);
    const double sample_outlier_dist_sq = sample_outlier_dist * sample_outlier_dist;
    for(int j=used_area.y;j<used_area.br().y-1;j++)
        for(int i=used_area.x;i<used_area.br().x-1;i++) {
            if (state(j,i) & (STATE_LOC_VALID|STATE_COORD_VALID)
                && state(j,i+1) & (STATE_LOC_VALID|STATE_COORD_VALID)
                && state(j+1,i) & (STATE_LOC_VALID|STATE_COORD_VALID)
                && state(j+1,i+1) & (STATE_LOC_VALID|STATE_COORD_VALID))
            {
            const cv::Vec3d& c00 = points(j,i);
            const cv::Vec3d& c01 = points(j,i+1);
            const cv::Vec3d& c10 = points(j+1,i);
            const cv::Vec3d& c11 = points(j+1,i+1);
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
                                cv::Vec3d c0 = (1-fx)*c00 + fx*c01;
                                cv::Vec3d c1 = (1-fx)*c10 + fx*c11;
                                cv::Vec3d expected = (1-fy)*c0 + fy*c1;
                                cv::Vec3d sample = SurfTrackerData::lookup_int_loc(sm,l);
                                const cv::Vec3d delta = sample - expected;
                                const int y = j*step+sy;
                                const int x = i*step+sx;
                                if (delta.dot(delta) <= sample_outlier_dist_sq) {
                                    points_hr(y,x) += sample;
                                    counts_hr(y,x) += 1;
                                }
                                else {
                                    rejected_hr(y,x) = 1;
                                }
                            }
                        }
                }
            }
            for(int sy=0;sy<=step;sy++)
                for(int sx=0;sx<=step;sx++) {
                    const int y = j*step+sy;
                    const int x = i*step+sx;
                    if (!counts_hr(y,x) && rejected_hr(y,x)) {
                        float fx = sx/step;
                        float fy = sy/step;
                        cv::Vec3d c0 = (1-fx)*c00 + fx*c01;
                        cv::Vec3d c1 = (1-fx)*c10 + fx*c11;
                        cv::Vec3d c = (1-fy)*c0 + fy*c1;
                        points_hr(y,x) = c;
                        counts_hr(y,x) = 1;
                    }
                }
            if (!counts_hr(j*step+1,i*step+1) && inpaint) {
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
    bool debug_images = false,
    int reopt_support_radius = 0,
    bool reopt_robust_affine_support = false,
    bool reopt_drop_unsupported_points = true)
{
    std::cout << "optimizer: optimizing surface " << state.size() << " " << used_area <<  " " << static_bounds << std::endl;
    reopt_support_radius = std::max(0, reopt_support_radius);
    std::cout << "optimizer: reopt_support_radius " << reopt_support_radius
              << " reopt_robust_affine_support " << reopt_robust_affine_support
              << " reopt_drop_unsupported_points " << reopt_drop_unsupported_points << std::endl;

    cv::Mat_<cv::Vec3d> points_new = points.clone();
    // QuadSurface destruction can release storage shared with optimizer helper
    // surfaces on this path. Keep these remap-only helpers alive past the
    // optimizer cleanup.
    QuadSurface* sm = new QuadSurface(points, {1,1});

    std::shared_mutex mutex;

    SurfTrackerData data_new;
    data_new._data = data._data;

    const cv::Rect requested_used_area = used_area;
    if ((requested_used_area & static_bounds) == requested_used_area) {
        std::cout << "optimizer: ignoring all-covering static bounds for active window "
                  << requested_used_area << std::endl;
        static_bounds = cv::Rect();
    }

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
                    const cv::Vec2d src_loc = data_inp.loc(sm_inp, {j,i});
                    auto add_support_surfs_at = [&](int yy, int xx) {
                        if (yy < 0 || yy >= state.rows || xx < 0 || xx >= state.cols)
                            return;
                        const auto& cell_surfs = data.surfsC({yy, xx});
                        surfs.insert(cell_surfs.begin(), cell_surfs.end());
                    };

                    if (reopt_support_radius > 0) {
                        for (int yy = y - reopt_support_radius; yy <= y + reopt_support_radius + 1; ++yy)
                            for (int xx = x - reopt_support_radius; xx <= x + reopt_support_radius + 1; ++xx)
                                add_support_surfs_at(yy, xx);
                    } else {
                        add_support_surfs_at(y, x);
                        add_support_surfs_at(y, x + 1);
                        add_support_surfs_at(y + 1, x);
                        add_support_surfs_at(y + 1, x + 1);
                    }

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
                        else if (reopt_robust_affine_support) {
                            std::vector<LocalSurfaceLocSample> loc_samples;
                            const int sample_radius = std::max(1, reopt_support_radius);
                            for (int yy = y - sample_radius; yy <= y + sample_radius + 1; ++yy) {
                                for (int xx = x - sample_radius; xx <= x + sample_radius + 1; ++xx) {
                                    if (yy < 0 || yy >= state.rows || xx < 0 || xx >= state.cols)
                                        continue;
                                    if ((state(yy, xx) & STATE_LOC_VALID) == 0 || !data.valid_int(s, {yy, xx}))
                                        continue;
                                    loc_samples.push_back({
                                        cv::Vec2d(static_cast<double>(yy) - src_loc[0],
                                                  static_cast<double>(xx) - src_loc[1]),
                                        data.loc(s, {yy, xx})
                                    });
                                }
                            }

                            if (!loc_samples.empty()) {
                                const cv::Vec2d predicted_loc = robust_affine_surface_loc_prediction(loc_samples);
                                const cv::Vec3d coord = SurfTrackerData::lookup_int_loc(
                                    s,
                                    cv::Vec2f(static_cast<float>(predicted_loc[0]),
                                              static_cast<float>(predicted_loc[1])));
                                if (coord[0] != -1 && cv::norm(coord - points_out(j, i)) <= same_surface_th) {
                                    mutex.lock();
                                    data_out.surfs({j,i}).insert(s);
                                    data_out.loc(s, {j,i}) = predicted_loc;
                                    mutex.unlock();
                                }
                            }
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
                    for(const auto& hit : surface_patch_hits(surface_patch_index, points_out(j, i))) {
                        QuadSurface* test_surf = hit.surface.get();
                        if (!test_surf) {
                            continue;
                        }
                        mutex.lock_shared();
                        if (data_out.has(test_surf, {j,i})) {
                            mutex.unlock();
                            continue;
                        }
                        mutex.unlock();

                        int count = 0;
                        cv::Vec3f loc_3d = test_surf->loc_raw(hit.ptr);
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

    if (reopt_drop_unsupported_points) {
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
    } else {
        std::cout << "optimizer: retaining unsupported points after reopt" << std::endl;
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
    bool bidirectional = params.value("bidirectional", false);
    int global_steps_per_window = params.value("global_steps_per_window", 1);


    std::cout << "global_steps_per_window: " << global_steps_per_window << std::endl;
    std::cout << "flip_x: " << flip_x << std::endl;
    std::cout << "bidirectional: " << bidirectional << std::endl;
    std::filesystem::path tgt_dir = params["tgt_dir"].get<std::string>();

    std::unordered_map<std::string,QuadSurface *> surfs;
    float src_step = params.value("src_step", 20);
    float step = params.value("step", 10);
    int max_width = params.value("max_width", 80000);
    int max_height = std::max(1, params.value("max_height", 50000));
    const bool debug_images = params.value("debug_images", false);
    const bool robust_affine_initial_guess = params.value("robust_affine_initial_guess", false);
    const int robust_affine_sample_radius = std::max(1, params.value("robust_affine_sample_radius", 1));
    const int reopt_support_radius = std::max(0, params.value("reopt_support_radius", 0));
    const bool reopt_robust_affine_support = params.value("reopt_robust_affine_support", false);
    const bool reopt_drop_unsupported_points = params.value("reopt_drop_unsupported_points", true);
    const bool reopt_drop_disconnected_components = params.value("reopt_drop_disconnected_components", false);
    const bool sweep_prune_distance_enabled = params.value("sweep_prune_distance_enabled", false);
    const std::string sweep_prune_target_path = params.value("sweep_prune_target_path", std::string{});
    const double sweep_prune_best_coverage_adjusted_distance_vx =
        params.value("sweep_prune_best_coverage_adjusted_distance_vx", std::numeric_limits<double>::infinity());
    const double sweep_prune_target_area_vx2 =
        params.value("sweep_prune_target_area_vx2", 0.0);
    const double sweep_prune_distance_margin =
        std::max(1.0, params.value("sweep_prune_distance_margin", 1.5));
    const double sweep_prune_search_radius_vx =
        std::max(1.0, params.value("sweep_prune_search_radius_vx", 1000.0));
    const int sweep_prune_min_generations = std::max(0, params.value("sweep_prune_min_generations", 1000));
    const int sweep_prune_interval_generations = std::max(1, params.value("sweep_prune_interval_generations", 10));
    const int sweep_prune_min_valid_cells = std::max(1, params.value("sweep_prune_min_valid_cells", 100));
    const int sweep_prune_max_samples = std::max(1, params.value("sweep_prune_max_samples", 2000));
    const int reopt_interval = std::max(0, params.value("reopt_interval", 0));

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
    inlier_threshold_drop_step = std::max(1, params.value("inlier_threshold_drop_step", 2));
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
    std::cout << "  inlier_threshold_drop_step: " << inlier_threshold_drop_step << std::endl;
    std::cout << "  sliding_w_scale: " << sliding_w_scale << std::endl;
    std::cout << "  z_loc_loss_w: " << z_loc_loss_w << std::endl;
    std::cout << "  dist_loss_2d_w: " << dist_loss_2d_w << std::endl;
    std::cout << "  dist_loss_3d_w: " << dist_loss_3d_w << std::endl;
    std::cout << "  max_width: " << max_width << std::endl;
    std::cout << "  max_height: " << max_height << std::endl;
    std::cout << "  robust_affine_initial_guess: " << robust_affine_initial_guess << std::endl;
    std::cout << "  robust_affine_sample_radius: " << robust_affine_sample_radius << std::endl;
    std::cout << "  reopt_support_radius: " << reopt_support_radius << std::endl;
    std::cout << "  reopt_robust_affine_support: " << reopt_robust_affine_support << std::endl;
    std::cout << "  reopt_drop_unsupported_points: " << reopt_drop_unsupported_points << std::endl;
    std::cout << "  reopt_drop_disconnected_components: " << reopt_drop_disconnected_components << std::endl;
    if (sweep_prune_distance_enabled) {
        std::cout << "  sweep_prune_distance: best_coverage_adjusted_vx=" << sweep_prune_best_coverage_adjusted_distance_vx
                  << " target=" << sweep_prune_target_path
                  << " target_area_vx2=" << sweep_prune_target_area_vx2
                  << " margin=" << sweep_prune_distance_margin
                  << " search_radius_vx=" << sweep_prune_search_radius_vx
                  << " min_generations=" << sweep_prune_min_generations << std::endl;
    }
    std::cout << "  reopt_interval: " << reopt_interval << std::endl;
    if (external_surface_patch_index) {
        std::cout << "  external_surface_patch_index: true" << std::endl;
    }
    if (enforce_z_range)
        std::cout << "  z_range: [" << z_min << ", " << z_max << "]" << std::endl;

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

    std::cout << "total surface count (after defective filter): " << surfs.size() << std::endl;
    std::cout << "seed " << seed << " name " << surface_name(seed) << std::endl;

    SurfacePatchIndex surface_patch_index;
    SurfacePatchIndex* surface_patch_index_ptr = external_surface_patch_index;
    if (!surface_patch_index_ptr) {
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
        const auto rebuild_start = std::chrono::steady_clock::now();
        surface_patch_index.rebuild(patch_surfaces, 0.0f);
        const double rebuild_elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - rebuild_start).count();
        std::cout << "SurfacePatchIndex rebuild time=" << rebuild_elapsed << "s" << std::endl;
        surface_patch_index_ptr = &surface_patch_index;
        std::cout << "SurfacePatchIndex built for " << patch_surfaces.size()
                  << " surfaces patches=" << surface_patch_index.patchCount()
                  << std::endl;
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
    int h = max_height/src_step/step*2+10+2*closing_r;
    const GrowthConfig growth_config = parse_growth_config(params, bidirectional, flip_x);
    growth_config.log(std::cout, stop_gen);

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
        const int extra_left = growth_config.grow_left ? grow_initial_extra_cols_lr : 0;
        const int extra_right = growth_config.grow_right ? grow_initial_extra_cols_lr : 0;
        const int extra_up = growth_config.grow_up ? grow_initial_extra_rows_lr : 0;
        const int extra_down = growth_config.grow_down ? grow_initial_extra_rows_lr : 0;
        expanded_left = extra_left;
        expanded_right = extra_right;
        expanded_up = extra_up;
        expanded_down = extra_down;
        max_extra_left = growth_config.grow_left ? grow_max_extra_cols_lr : 0;
        max_extra_right = growth_config.grow_right ? grow_max_extra_cols_lr : 0;
        max_extra_up = growth_config.grow_up ? grow_max_extra_rows_lr : 0;
        max_extra_down = growth_config.grow_down ? grow_max_extra_rows_lr : 0;
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
        growth_config.disable_grid_expansion && resume_growth && resume_grid_bounds.area() > 0;

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

    bool sweep_prune_distance_active = false;
    std::shared_ptr<QuadSurface> sweep_prune_target_surface;
    SurfacePatchIndex sweep_prune_target_index;
    if (sweep_prune_distance_enabled &&
        !sweep_prune_target_path.empty() &&
        std::isfinite(sweep_prune_best_coverage_adjusted_distance_vx) &&
        sweep_prune_target_area_vx2 > 0.0) {
        try {
            sweep_prune_target_surface = std::make_shared<QuadSurface>(std::filesystem::path(sweep_prune_target_path));
            sweep_prune_target_surface->ensureLoaded();
            sweep_prune_target_index.setReadOnly(true);
            sweep_prune_target_index.rebuild({sweep_prune_target_surface}, 0.0f);
            sweep_prune_distance_active = !sweep_prune_target_index.empty();
            if (sweep_prune_distance_active) {
                std::cout << "sweep distance prune target indexed: "
                          << sweep_prune_target_path
                          << " patches=" << sweep_prune_target_index.patchCount() << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "sweep distance prune disabled: failed to index target "
                      << sweep_prune_target_path << ": " << e.what() << std::endl;
        }
    }

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

            int fringe_count = 0;
            int discovered_patch_ref_count = 0;
            for (int j = std::max(0, used_area.y - 1); j <= std::min(used_area.br().y + 1, state.rows - 1); ++j) {
                for (int i = std::max(0, used_area.x - 1); i <= std::min(used_area.br().x + 1, state.cols - 1); ++i) {
                    const cv::Vec2i p(j, i);
                    if ((state(p) & STATE_LOC_VALID) == 0) {
                        continue;
                    }
                    bool has_invalid_neighbor = false;
                    for (const auto& n : growth_config.neighs) {
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
                    discovered_patch_ref_count += add_surface_patch_candidates(
                        p, points(p), data, surface_patch_index_ptr, seed);
                }
            }

            int seeded_patch_ref_locations = 0;
            int seeded_patch_refs = 0;
            for (int j = std::max(0, used_area.y); j <= std::min(used_area.br().y, state.rows - 1); ++j) {
                for (int i = std::max(0, used_area.x); i <= std::min(used_area.br().x, state.cols - 1); ++i) {
                    const cv::Vec2i p(j, i);
                    if ((state(p) & STATE_LOC_VALID) == 0) {
                        continue;
                    }

                    const int added = add_surface_patch_candidates(
                        p, points(p), data, surface_patch_index_ptr, seed);
                    if (added > 0) {
                        seeded_patch_ref_locations++;
                        seeded_patch_refs += added;
                    }
                }
            }

            std::cout << "resume_growth initialized " << resumed_count
                      << " low-res points, fringe " << fringe_count
                      << ", patch-index refs discovered " << discovered_patch_ref_count
                      << ", seeded patch-index refs " << seeded_patch_refs
                      << " at " << seeded_patch_ref_locations << " points"
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
            cv::Vec3d coord = points(p);
            int added = add_surface_patch_candidates(p, coord, data, surface_patch_index_ptr, seed);
            std::cout << "testing " << p << " from patch-index cands: " << added << coord << std::endl;
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
    bool suppress_empty_fringe_reopt_after_component_drop = false;
    int component_drop_valid_count = 0;
    bool component_drop_reopt_skip_logged = false;
    const int configured_consensus_limit_th = params.value("consensus_limit_th", 2);
    int emergency_consensus_limit_th = configured_consensus_limit_th;

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
    int last_expansion_loc_valid_count = best_loc_valid_count;
    int no_growth_expansions = 0;

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

    auto count_all_valid_neighbors = [&](const cv::Vec2i& p) {
        int count = 0;
        for (const auto& n : growth_config.all_8_neighs) {
            const cv::Vec2i pn = p + n;
            if (pn[0] >= 0 && pn[0] < state.rows &&
                pn[1] >= 0 && pn[1] < state.cols &&
                (state(pn) & STATE_LOC_VALID)) {
                count++;
            }
        }
        return count;
    };

    auto reseed_fringe_from_valid = [&](const cv::Rect& area) {
        const cv::Rect active = area & cv::Rect(0, 0, state.cols, state.rows);
        if (active.area() <= 0) {
            return;
        }
        for(int j=std::max(0, active.y-2);j<=std::min(active.br().y+2, state.rows-1);j++)
            for(int i=std::max(0, active.x-2);i<=std::min(active.br().x+2, state.cols-1);i++)
                if (state(j,i) & STATE_LOC_VALID)
                    fringe.insert(cv::Vec2i(j,i));
    };

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
        const int succ_before_generation = succ;
        std::unordered_set<cv::Vec2i,vec2i_hash> cands;
        if (generation == 0 && !initialized_from_resume) {
            cands.insert(cv::Vec2i(y0-1,x0));
        }
        else
            for(const auto& p : fringe)
            {
                if ((state(p) & STATE_LOC_VALID) == 0)
                    continue;

                for(const auto& n : growth_config.neighs) {
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
                    if (!save_bounds_inv.contains(cv::Point(pn)) && save_bounds_inv.br().y <= pn[1]) {
                        at_right_border = true;
                    }
                }
            }
        fringe.clear();

        std::cout << "go with cands " << cands.size() << " inl_th " << curr_best_inl_th << std::endl;

        const std::vector<cv::Vec2i> candidate_points(cands.begin(), cands.end());
        auto existing_support_depth = [&](const cv::Vec2i& p) {
            if (!growth_config.candidate_priority_existing_depth) {
                return 0;
            }
            int best_depth = 0;
            for (const cv::Vec2i& dir : growth_config.legacy_4_neighs) {
                int depth = 0;
                for (int step_idx = 1; step_idx <= growth_config.candidate_support_depth_radius; ++step_idx) {
                    const cv::Vec2i q = p + step_idx * dir;
                    if (q[0] < 0 || q[0] >= state.rows ||
                        q[1] < 0 || q[1] >= state.cols ||
                        (state(q) & STATE_LOC_VALID) == 0) {
                        break;
                    }
                    depth++;
                }
                best_depth = std::max(best_depth, depth);
            }
            return best_depth;
        };
        GrowthCandidateOrdering candidate_ordering = order_growth_candidates(
            candidate_points,
            growth_config,
            count_all_valid_neighbors,
            existing_support_depth);
        struct RolloutProbe {
            bool accepted = false;
            cv::Vec2i p{-1, -1};
            cv::Vec3d coord{-1, -1, -1};
            QuadSurface* surf = nullptr;
            cv::Vec2d loc{-1, -1};
            int inliers = -1;
            bool ref_seed = false;
            bool approved = false;
        };
        struct RolloutResult {
            int64_t score = std::numeric_limits<int64_t>::min();
            std::vector<RolloutProbe> accepted_probes;
        };
        auto probe_candidate = [&](const cv::Vec2i& p,
                                   SurfTrackerData& probe_data,
                                   cv::Mat_<uint8_t>& probe_state,
                                   cv::Mat_<cv::Vec3d>& probe_points,
                                   const cv::Rect& probe_used_area,
                                   bool allow_approved_log) {
            RolloutProbe probe;
            probe.p = p;
            if (p[0] < 0 || p[0] >= probe_state.rows || p[1] < 0 || p[1] >= probe_state.cols)
                return probe;
            if (probe_state(p) & STATE_LOC_VALID)
                return probe;
            if (probe_points(p)[0] != -1)
                return probe;

            constexpr int r = 1;
            std::vector<QuadSurface*> local_surfs;
            local_surfs.reserve(8);
            auto add_local_surf = [&](QuadSurface* surf) {
                if (surf && std::find(local_surfs.begin(), local_surfs.end(), surf) == local_surfs.end()) {
                    local_surfs.push_back(surf);
                }
            };
            add_local_surf(seed);
            for(int oy=std::max(p[0]-r,0);oy<=std::min(p[0]+r,h-1);oy++)
                for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,w-1);ox++)
                    if (probe_state(oy,ox) & STATE_LOC_VALID) {
                        auto p_surfs = probe_data.surfsC({oy,ox});
                        for (auto* surf : p_surfs) {
                            add_local_surf(surf);
                        }
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
                std::vector<LocalSurfaceLocSample> loc_samples;
                bool ref_seed = false;
                const int sample_r = robust_affine_initial_guess ? robust_affine_sample_radius : r;
                for(int oy=std::max(p[0]-sample_r,0);oy<=std::min(p[0]+sample_r,h-1);oy++)
                    for(int ox=std::max(p[1]-sample_r,0);ox<=std::min(p[1]+sample_r,w-1);ox++)
                        if ((probe_state(oy,ox) & STATE_LOC_VALID) && probe_data.valid_int(ref_surf,{oy,ox})) {
                            const cv::Vec2d loc = probe_data.loc(ref_surf,{oy,ox});
                            ref_count++;
                            avg += loc;
                            loc_samples.push_back({
                                cv::Vec2d(static_cast<double>(oy - p[0]),
                                          static_cast<double>(ox - p[1])),
                                loc
                            });
                            if (probe_data.seed_loc == cv::Vec2i(oy,ox))
                                ref_seed = true;
                        }

                if (ref_count < 2 && !ref_seed)
                    continue;

                avg /= ref_count;
                const uint8_t probe_state_old = probe_state(p);
                probe_data.loc(ref_surf,p) = robust_affine_initial_guess
                    ? robust_affine_surface_loc_prediction(loc_samples)
                    : avg + deterministic_probe_jitter(p);
                probe_state(p) = STATE_LOC_VALID | STATE_COORD_VALID;

                ceres::Problem problem;
                int straight_count_init = 0;
                int count_init = surftrack_add_local(ref_surf, p, probe_data, problem, probe_state, probe_points, step, src_step, LOSS_ZLOC, &straight_count_init);
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);

                bool fail = false;
                cv::Vec2d ref_loc = probe_data.loc(ref_surf,p);
                if (!probe_data.valid_int(ref_surf,p))
                    fail = true;

                cv::Vec3d coord;
                if (!fail) {
                    coord = probe_data.lookup_int(ref_surf,p);
                    if (coord[0] == -1)
                        fail = true;
                }

                if (fail) {
                    probe_state(p) = probe_state_old;
                    probe_data.erase(ref_surf, p);
                    continue;
                }

                probe_state(p) = probe_state_old;
                int inliers_sum = 0;
                int inliers_count = 0;

                if (approved_sm.contains(ref_surf) && straight_count_init >= 2 && count_init >= 4) {
                    if (enforce_z_range && (coord[2] < z_min || coord[2] > z_max)) {
                        probe_data.erase(ref_surf, p);
                        continue;
                    }
                    if (allow_approved_log && used_approved_names.insert(ref_surf->id).second) {
                        approved_log << ref_surf->id << std::endl;
                        approved_log.flush();
                    }
                    best_inliers = 1000;
                    best_coord = coord;
                    best_surf = ref_surf;
                    best_loc = ref_loc;
                    best_ref_seed = ref_seed;
                    best_approved = true;
                    probe_data.erase(ref_surf, p);
                    break;
                }

                for(auto test_surf : local_surfs) {
                    auto ptr = test_surf->pointer();
                    if (test_surf->pointTo(ptr, coord, same_surface_th, 10,
                                                       surface_patch_index_ptr) <= same_surface_th) {
                        int count = 0;
                        int straight_count = 0;
                        probe_state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
                        cv::Vec3f loc = test_surf->loc_raw(ptr);
                        probe_data.loc(test_surf, p) = {loc[1], loc[0]};
                        float cost = local_cost(test_surf, p, probe_data, probe_state, probe_points, step, src_step, &count, &straight_count, &coord);
                        probe_state(p) = probe_state_old;
                        probe_data.erase(test_surf, p);
                        if (cost < local_cost_inl_th && (ref_seed || (count >= 2 && straight_count >= straight_min_count))) {
                            inliers_sum += count;
                            inliers_count++;
                        }
                    }
                }
                if ((inliers_count >= 2 || ref_seed) && inliers_sum > best_inliers) {
                    if (enforce_z_range && (coord[2] < z_min || coord[2] > z_max)) {
                        probe_data.erase(ref_surf, p);
                        continue;
                    }
                    best_inliers = inliers_sum;
                    best_coord = coord;
                    best_surf = ref_surf;
                    best_loc = ref_loc;
                    best_ref_seed = ref_seed;
                    best_approved = false;
                }
                probe_data.erase(ref_surf, p);
            }

            if (best_inliers >= curr_best_inl_th || best_ref_seed) {
                if (enforce_z_range && (best_coord[2] < z_min || best_coord[2] > z_max)) {
                    return probe;
                }
                cv::Vec2f tmp_loc_;
                if (probe_used_area.width > 3 && probe_used_area.height > 3) {
                    float dist = pointTo(tmp_loc_, probe_points(probe_used_area), best_coord, same_surface_th, 1000, 1.0/(step*src_step));
                    tmp_loc_ += cv::Vec2f(probe_used_area.x,probe_used_area.y);
                    if (dist <= same_surface_th) {
                        int state_sum = probe_state(tmp_loc_[1],tmp_loc_[0]) + probe_state(tmp_loc_[1]+1,tmp_loc_[0]) + probe_state(tmp_loc_[1],tmp_loc_[0]+1) + probe_state(tmp_loc_[1]+1,tmp_loc_[0]+1);
                        if (state_sum)
                            return probe;
                    }
                }
                probe.accepted = true;
                probe.coord = best_coord;
                probe.surf = best_surf;
                probe.loc = best_loc;
                probe.inliers = best_inliers;
                probe.ref_seed = best_ref_seed;
                probe.approved = best_approved;
            }
            return probe;
        };
        auto commit_rollout_probe_speculative = [&](const RolloutProbe& probe,
                                                    SurfTrackerData& rollout_data,
                                                    cv::Mat_<uint8_t>& rollout_state,
                                                    cv::Mat_<cv::Vec3d>& rollout_points,
                                                    cv::Rect& rollout_used_area,
                                                    std::vector<cv::Vec2i>& rollout_frontier) {
            if (!probe.accepted || !probe.surf)
                return false;
            rollout_data.surfs(probe.p).insert(probe.surf);
            rollout_data.loc(probe.surf, probe.p) = probe.loc;
            rollout_state(probe.p) = STATE_LOC_VALID | STATE_COORD_VALID;
            rollout_points(probe.p) = probe.coord;
            for (auto hit : surface_patch_hits(surface_patch_index_ptr, probe.coord, probe.surf)) {
                QuadSurface* s = hit.surface.get();
                if (!s)
                    continue;
                cv::Vec3f loc = s->loc_raw(hit.ptr);
                cv::Vec3f coord = SurfTrackerData::lookup_int_loc(s, {loc[1], loc[0]});
                if (coord[0] == -1)
                    continue;
                int count = 0;
                float cost = local_cost_destructive(s, probe.p, rollout_data, rollout_state, rollout_points, step, src_step, loc, &count);
                if (cost < local_cost_inl_th) {
                    rollout_data.loc(s, probe.p) = {loc[1], loc[0]};
                    rollout_data.surfs(probe.p).insert(s);
                }
            }
            if (!rollout_used_area.contains(cv::Point(probe.p[1], probe.p[0]))) {
                rollout_used_area = rollout_used_area | cv::Rect(probe.p[1],probe.p[0],1,1);
            }
            rollout_frontier.push_back(probe.p);
            return true;
        };
        auto commit_rollout_probe_final = [&](const RolloutProbe& probe,
                                              std::vector<cv::Vec2i>& rollout_frontier) {
            if (!probe.accepted || !probe.surf || (state(probe.p) & STATE_LOC_VALID))
                return false;
            if (probe.coord[0] == -1)
                throw std::runtime_error("oops rollout probe coord[0]");

            constexpr int r = 1;
            std::vector<QuadSurface*> local_surfs;
            local_surfs.reserve(8);
            auto add_local_surf = [&](QuadSurface* surf) {
                if (surf && std::find(local_surfs.begin(), local_surfs.end(), surf) == local_surfs.end()) {
                    local_surfs.push_back(surf);
                }
            };
            add_local_surf(seed);
            for(int oy=std::max(probe.p[0]-r,0);oy<=std::min(probe.p[0]+r,h-1);oy++)
                for(int ox=std::max(probe.p[1]-r,0);ox<=std::min(probe.p[1]+r,w-1);ox++)
                    if (state(oy,ox) & STATE_LOC_VALID) {
                        auto p_surfs = data.surfsC({oy,ox});
                        for (auto* surf : p_surfs) {
                            add_local_surf(surf);
                        }
                    }

            if (probe.approved && used_approved_names.insert(probe.surf->id).second) {
                std::cout << "found approved sm " << probe.surf->id << std::endl;
                approved_log << probe.surf->id << std::endl;
                approved_log.flush();
            }

            data.surfs(probe.p).insert(probe.surf);
            data.loc(probe.surf, probe.p) = probe.loc;
            state(probe.p) = STATE_LOC_VALID | STATE_COORD_VALID;
            points(probe.p) = probe.coord;

            ceres::Problem problem;
            surftrack_add_local(probe.surf, probe.p, data, problem, state, points, step, src_step, SURF_LOSS | LOSS_ZLOC);

            std::vector<SurfacePatchIndex::LookupResult> more_local_hits;
            for(auto test_surf : local_surfs) {
                if (test_surf == probe.surf)
                    continue;

                auto ptr = test_surf->pointer();
                if (test_surf->pointTo(ptr, probe.coord, same_surface_th, 10,
                                                   surface_patch_index_ptr) <= same_surface_th) {
                    cv::Vec3f loc = test_surf->loc_raw(ptr);
                    data.loc(test_surf, probe.p) = {loc[1], loc[0]};
                    int count = 0;
                    float cost = local_cost(test_surf, probe.p, data, state, points, step, src_step, &count);
                    if (cost < local_cost_inl_th) {
                        data.surfs(probe.p).insert(test_surf);
                        surftrack_add_local(test_surf, probe.p, data, problem, state, points, step, src_step, SURF_LOSS | LOSS_ZLOC);
                    }
                    else {
                        data.erase(test_surf, probe.p);
                    }
                }
            }

            for (auto hit : surface_patch_hits(surface_patch_index_ptr, probe.coord, probe.surf)) {
                QuadSurface* s = hit.surface.get();
                if (s && std::find(local_surfs.begin(), local_surfs.end(), s) == local_surfs.end()) {
                    more_local_hits.push_back(std::move(hit));
                }
            }

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            for(const auto& hit : more_local_hits) {
                QuadSurface* test_surf = hit.surface.get();
                if (!test_surf) {
                    continue;
                }
                cv::Vec3f loc = test_surf->loc_raw(hit.ptr);
                cv::Vec3f coord = SurfTrackerData::lookup_int_loc(test_surf, {loc[1], loc[0]});
                if (coord[0] == -1) {
                    continue;
                }
                int count = 0;
                float cost = local_cost_destructive(test_surf, probe.p, data, state, points, step, src_step, loc, &count);
                if (cost < local_cost_inl_th) {
                    data.loc(test_surf, probe.p) = {loc[1], loc[0]};
                    data.surfs(probe.p).insert(test_surf);
                }
            }

            if (!used_area.contains(cv::Point(probe.p[1], probe.p[0]))) {
                used_area = used_area | cv::Rect(probe.p[1],probe.p[0],1,1);
            }
            rollout_frontier.push_back(probe.p);
            return true;
        };
        auto completes_existing_l_shape = [&](const cv::Vec2i& p) {
            const cv::Vec2i block_origins[] = {
                p,
                p + cv::Vec2i{-1, 0},
                p + cv::Vec2i{0, -1},
                p + cv::Vec2i{-1, -1},
            };
            for (const cv::Vec2i& origin : block_origins) {
                int valid_count = 0;
                bool contains_p = false;
                for (int dy = 0; dy <= 1; ++dy) {
                    for (int dx = 0; dx <= 1; ++dx) {
                        const cv::Vec2i q = origin + cv::Vec2i{dy, dx};
                        if (q == p) {
                            contains_p = true;
                            continue;
                        }
                        if (q[0] < 0 || q[0] >= state.rows ||
                            q[1] < 0 || q[1] >= state.cols ||
                            (state(q) & STATE_LOC_VALID) == 0) {
                            valid_count = -1;
                            break;
                        }
                        valid_count++;
                    }
                    if (valid_count < 0) {
                        break;
                    }
                }
                if (contains_p && valid_count == 3) {
                    return true;
                }
            }
            return false;
        };
        auto rollout_score_root = [&](const cv::Vec2i& root) {
            RolloutResult result;
            SurfTrackerData rollout_data = data;
            cv::Mat_<uint8_t> rollout_state = state.clone();
            cv::Mat_<cv::Vec3d> rollout_points = points.clone();
            cv::Rect rollout_used_area = used_area;
            std::vector<cv::Vec2i> frontier;
            int64_t accepted_count = 0;
            int64_t inlier_sum = 0;
            int64_t connection_sum = 0;
            int64_t base_connection_sum = 0;
            int64_t internal_connection_sum = 0;
            std::vector<cv::Vec2i> rollout_accepted;
            rollout_accepted.reserve(static_cast<std::size_t>(
                1 + growth_config.rollout_depth * growth_config.rollout_max_children));
            auto rollout_valid_neighbors = [&](const cv::Vec2i& p) {
                int count = 0;
                for (const auto& n : growth_config.all_8_neighs) {
                    const cv::Vec2i pn = p + n;
                    if (pn[0] >= 0 && pn[0] < rollout_state.rows &&
                        pn[1] >= 0 && pn[1] < rollout_state.cols &&
                        (rollout_state(pn) & STATE_LOC_VALID)) {
                        count++;
                    }
                }
                return count;
            };
            auto rollout_base_neighbors = [&](const cv::Vec2i& p) {
                int count = 0;
                for (const auto& n : growth_config.all_8_neighs) {
                    const cv::Vec2i pn = p + n;
                    if (pn[0] >= 0 && pn[0] < state.rows &&
                        pn[1] >= 0 && pn[1] < state.cols &&
                        (state(pn) & STATE_LOC_VALID)) {
                        count++;
                    }
                }
                return count;
            };
            auto rollout_internal_neighbors = [&](const cv::Vec2i& p) {
                int count = 0;
                for (const auto& n : growth_config.all_8_neighs) {
                    const cv::Vec2i pn = p + n;
                    if (std::find(rollout_accepted.begin(), rollout_accepted.end(), pn) != rollout_accepted.end()) {
                        count++;
                    }
                }
                return count;
            };

            RolloutProbe root_probe = probe_candidate(root, rollout_data, rollout_state, rollout_points, rollout_used_area, false);
            if (!root_probe.accepted)
                return result;
            if (!commit_rollout_probe_speculative(root_probe, rollout_data, rollout_state, rollout_points, rollout_used_area, frontier))
                return result;
            result.accepted_probes.push_back(root_probe);
            accepted_count++;
            inlier_sum += std::max(0, root_probe.inliers);
            rollout_accepted.push_back(root);
            connection_sum += static_cast<int>(rollout_data.surfs(root).size()) + rollout_valid_neighbors(root);
            base_connection_sum += rollout_base_neighbors(root);
            internal_connection_sum += rollout_internal_neighbors(root);

            for (int depth = 1; depth < growth_config.rollout_depth && !frontier.empty(); ++depth) {
                struct RolloutCandidate {
                    cv::Vec2i p;
                    int valid_neighbors = 0;
                };
                std::vector<RolloutCandidate> next_candidates;
                std::vector<cv::Vec2i> seen_next;
                next_candidates.reserve(frontier.size() * growth_config.neighs.size());
                seen_next.reserve(frontier.size() * growth_config.neighs.size());
                for (const cv::Vec2i& p : frontier) {
                    for (const cv::Vec2i& n : growth_config.neighs) {
                        const cv::Vec2i pn = p + n;
                        if (!save_bounds_inv.contains(cv::Point(pn)) ||
                            (rollout_state(pn) & STATE_LOC_VALID) ||
                            std::find(seen_next.begin(), seen_next.end(), pn) != seen_next.end()) {
                            continue;
                        }
                        seen_next.push_back(pn);
                        next_candidates.push_back({pn, rollout_valid_neighbors(pn)});
                    }
                }
                std::sort(next_candidates.begin(), next_candidates.end(), [](const RolloutCandidate& a, const RolloutCandidate& b) {
                    if (a.valid_neighbors != b.valid_neighbors)
                        return a.valid_neighbors > b.valid_neighbors;
                    if (a.p[0] != b.p[0])
                        return a.p[0] < b.p[0];
                    return a.p[1] < b.p[1];
                });
                if (static_cast<int>(next_candidates.size()) > growth_config.rollout_max_children) {
                    next_candidates.resize(static_cast<std::size_t>(growth_config.rollout_max_children));
                }
                std::vector<cv::Vec2i> next_frontier;
                for (const RolloutCandidate& candidate : next_candidates) {
                    RolloutProbe probe = probe_candidate(candidate.p, rollout_data, rollout_state, rollout_points, rollout_used_area, false);
                    if (!commit_rollout_probe_speculative(probe, rollout_data, rollout_state, rollout_points, rollout_used_area, next_frontier))
                        continue;
                    result.accepted_probes.push_back(probe);
                    accepted_count++;
                    inlier_sum += std::max(0, probe.inliers);
                    rollout_accepted.push_back(candidate.p);
                    connection_sum += static_cast<int>(rollout_data.surfs(candidate.p).size()) + rollout_valid_neighbors(candidate.p);
                    base_connection_sum += rollout_base_neighbors(candidate.p);
                    internal_connection_sum += rollout_internal_neighbors(candidate.p);
                }
                frontier = std::move(next_frontier);
            }

            result.score = accepted_count * growth_config.rollout_area_weight +
                           inlier_sum * growth_config.rollout_inlier_weight +
                           connection_sum * growth_config.rollout_connection_weight +
                           base_connection_sum * growth_config.rollout_base_connection_weight +
                           internal_connection_sum * growth_config.rollout_internal_connection_weight;
            return result;
        };
        auto reset_candidate_processing = [&]() {
            for (const cv::Vec2i& p : candidate_ordering.points) {
                if (!(state(p) & STATE_LOC_VALID)) {
                    state(p) = static_cast<uint8_t>(state(p) & ~STATE_PROCESSING);
                }
            }
        };
        const bool rollout_mode_active = growth_config.rollout_growth &&
                                         generation >= rollout_warmup_generations &&
                                         !candidate_ordering.points.empty();
        if (rollout_mode_active) {
            std::vector<cv::Vec2i> rollout_roots;
            const std::size_t root_count = std::min<std::size_t>(
                candidate_ordering.points.size(),
                static_cast<std::size_t>(growth_config.rollout_width));
            rollout_roots.reserve(root_count);
            if (root_count == candidate_ordering.points.size()) {
                rollout_roots = candidate_ordering.points;
            } else {
                std::vector<uint8_t> selected(candidate_ordering.points.size(), 0);
                std::vector<int> nearest_dist_sq(candidate_ordering.points.size(), std::numeric_limits<int>::max());
                auto select_root = [&](std::size_t idx) {
                    selected[idx] = 1;
                    const cv::Vec2i root = candidate_ordering.points[idx];
                    rollout_roots.push_back(root);
                    for (std::size_t cand_idx = 0; cand_idx < candidate_ordering.points.size(); ++cand_idx) {
                        if (selected[cand_idx]) {
                            nearest_dist_sq[cand_idx] = 0;
                            continue;
                        }
                        const cv::Vec2i delta = candidate_ordering.points[cand_idx] - root;
                        nearest_dist_sq[cand_idx] = std::min(nearest_dist_sq[cand_idx], delta.dot(delta));
                    }
                };

                select_root(0);
                while (rollout_roots.size() < root_count) {
                    std::size_t best_idx = std::numeric_limits<std::size_t>::max();
                    int best_dist_sq = -1;
                    for (std::size_t idx = 0; idx < candidate_ordering.points.size(); ++idx) {
                        if (selected[idx]) {
                            continue;
                        }
                        if (nearest_dist_sq[idx] > best_dist_sq) {
                            best_dist_sq = nearest_dist_sq[idx];
                            best_idx = idx;
                        }
                    }
                    if (best_idx == std::numeric_limits<std::size_t>::max()) {
                        break;
                    }
                    select_root(best_idx);
                }
            }
            std::vector<RolloutResult> rollout_results(rollout_roots.size());
#pragma omp parallel for schedule(dynamic)
            for (int root_idx = 0; root_idx < static_cast<int>(rollout_roots.size()); ++root_idx) {
                rollout_results[static_cast<std::size_t>(root_idx)] = rollout_score_root(rollout_roots[static_cast<std::size_t>(root_idx)]);
            }
            struct RankedRollout {
                std::size_t root_idx = 0;
                int64_t score = std::numeric_limits<int64_t>::min();
            };
            std::vector<RankedRollout> ranked_rollouts;
            ranked_rollouts.reserve(rollout_roots.size());
            for (std::size_t root_idx = 0; root_idx < rollout_roots.size(); ++root_idx) {
                const RolloutResult& rollout = rollout_results[root_idx];
                if (rollout.accepted_probes.size() == 1 &&
                    !completes_existing_l_shape(rollout.accepted_probes.front().p)) {
                    continue;
                }
                if (rollout.score == std::numeric_limits<int64_t>::min()) {
                    continue;
                }
                ranked_rollouts.push_back({root_idx, rollout.score});
            }
            std::sort(ranked_rollouts.begin(), ranked_rollouts.end(),
                [&](const RankedRollout& a, const RankedRollout& b) {
                    if (a.score != b.score)
                        return a.score > b.score;
                    const cv::Vec2i& root_a = rollout_roots[a.root_idx];
                    const cv::Vec2i& root_b = rollout_roots[b.root_idx];
                    if (root_a[0] != root_b[0])
                        return root_a[0] < root_b[0];
                    return root_a[1] < root_b[1];
                });
            if (!ranked_rollouts.empty()) {
                struct SelectedRollout {
                    cv::Vec2i root;
                    int64_t score = std::numeric_limits<int64_t>::min();
                    std::vector<RolloutProbe> probes;
                };
                std::vector<SelectedRollout> selected_rollouts;
                selected_rollouts.reserve(static_cast<std::size_t>(
                    std::min(growth_config.rollout_max_commits_per_generation,
                             static_cast<int>(ranked_rollouts.size()))));
                std::vector<cv::Vec2i> selected_probe_points;
                const int min_separation_sq = growth_config.rollout_min_separation *
                                              growth_config.rollout_min_separation;
                auto rollout_is_separated = [&](const RolloutResult& rollout) {
                    if (min_separation_sq <= 0) {
                        return true;
                    }
                    for (const RolloutProbe& probe : rollout.accepted_probes) {
                        for (const cv::Vec2i& selected_p : selected_probe_points) {
                            const cv::Vec2i delta = probe.p - selected_p;
                            if (delta.dot(delta) < min_separation_sq) {
                                return false;
                            }
                        }
                    }
                    return true;
                };
                for (const RankedRollout& ranked : ranked_rollouts) {
                    if (static_cast<int>(selected_rollouts.size()) >=
                        growth_config.rollout_max_commits_per_generation) {
                        break;
                    }
                    const RolloutResult& rollout = rollout_results[ranked.root_idx];
                    if (!rollout_is_separated(rollout)) {
                        continue;
                    }
                    for (const RolloutProbe& probe : rollout.accepted_probes) {
                        selected_probe_points.push_back(probe.p);
                    }
                    selected_rollouts.push_back({
                        rollout_roots[ranked.root_idx],
                        ranked.score,
                        rollout.accepted_probes,
                    });
                }
                std::cout << "rollout selected " << selected_rollouts.size()
                          << " of " << ranked_rollouts.size()
                          << " valid rollouts from " << rollout_roots.size()
                          << " roots" << std::endl;
                reset_candidate_processing();
                int total_rollout_committed = 0;
                for (const SelectedRollout& selected_rollout : selected_rollouts) {
                    std::cout << "rollout selected root " << selected_rollout.root
                              << " score " << selected_rollout.score
                              << " with " << selected_rollout.probes.size()
                              << " accepted points" << std::endl;
                    SurfTrackerData rollout_revalidate_data = data;
                    cv::Mat_<uint8_t> rollout_revalidate_state = state.clone();
                    cv::Mat_<cv::Vec3d> rollout_revalidate_points = points.clone();
                    cv::Rect rollout_revalidate_used_area = used_area;
                    std::vector<cv::Vec2i> rollout_revalidate_frontier;
                    std::vector<RolloutProbe> revalidated_rollout_probes;
                    revalidated_rollout_probes.reserve(selected_rollout.probes.size());
                    for (const RolloutProbe& probe : selected_rollout.probes) {
                        RolloutProbe revalidated_probe = probe_candidate(probe.p,
                                                                         rollout_revalidate_data,
                                                                         rollout_revalidate_state,
                                                                         rollout_revalidate_points,
                                                                         rollout_revalidate_used_area,
                                                                         false);
                        if (commit_rollout_probe_speculative(revalidated_probe,
                                                             rollout_revalidate_data,
                                                             rollout_revalidate_state,
                                                             rollout_revalidate_points,
                                                             rollout_revalidate_used_area,
                                                             rollout_revalidate_frontier)) {
                            revalidated_rollout_probes.push_back(revalidated_probe);
                        }
                    }
                    if (revalidated_rollout_probes.size() == 1 &&
                        !completes_existing_l_shape(revalidated_rollout_probes.front().p)) {
                        std::cout << "rollout rejected one non-L point after live revalidation" << std::endl;
                        revalidated_rollout_probes.clear();
                    }
                    std::vector<cv::Vec2i> rollout_frontier_committed;
                    int rollout_committed = 0;
                    for (const RolloutProbe& probe : revalidated_rollout_probes) {
                        RolloutProbe live_probe = probe_candidate(probe.p, data, state, points, used_area, false);
                        if (!commit_rollout_probe_final(live_probe, rollout_frontier_committed))
                            continue;
                        generations(live_probe.p) = static_cast<uint16_t>(std::min(generation + 1, static_cast<int>(std::numeric_limits<uint16_t>::max())));
                        inliers_sum_dbg(live_probe.p) = live_probe.inliers;
                        for(int t=0;t<omp_get_max_threads();t++)
                            added_points_threads[t].push_back(live_probe.p);
                        fringe.insert(live_probe.p);
                        succ++;
                        rollout_committed++;
                    }
                    total_rollout_committed += rollout_committed;
                    if (rollout_committed != static_cast<int>(selected_rollout.probes.size())) {
                        std::cout << "rollout committed " << rollout_committed
                                  << " of " << selected_rollout.probes.size()
                                  << " after live revalidation" << std::endl;
                    }
                }
                if (total_rollout_committed > 0) {
                    used_area_hr = scaled_rect_trunc(used_area, step);
                }
            }
            else {
                reset_candidate_processing();
            }
            candidate_ordering.points.clear();
        }
        int best_inliers_gen = 0;
        OmpThreadPointCol threadcol(3, candidate_ordering.points);

        std::shared_mutex mutex;
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
                std::vector<LocalSurfaceLocSample> loc_samples;
                bool ref_seed = false;
                const int sample_r = robust_affine_initial_guess ? robust_affine_sample_radius : r;
                for(int oy=std::max(p[0]-sample_r,0);oy<=std::min(p[0]+sample_r,h-1);oy++)
                    for(int ox=std::max(p[1]-sample_r,0);ox<=std::min(p[1]+sample_r,w-1);ox++)
                        if ((state(oy,ox) & STATE_LOC_VALID) && data_th.valid_int(ref_surf,{oy,ox})) {
                            const cv::Vec2d loc = data_th.loc(ref_surf,{oy,ox});
                            ref_count++;
                            avg += loc;
                            loc_samples.push_back({
                                cv::Vec2d(static_cast<double>(oy - p[0]),
                                          static_cast<double>(ox - p[1])),
                                loc
                            });
                            if (data_th.seed_loc == cv::Vec2i(oy,ox))
                                ref_seed = true;
                        }

                if (ref_count < 2 && !ref_seed)
                    continue;

                avg /= ref_count;

                data_th.loc(ref_surf,p) = robust_affine_initial_guess
                    ? robust_affine_surface_loc_prediction(loc_samples)
                    : avg + cv::Vec2d((rand() % 1000)/500.0-1, (rand() % 1000)/500.0-1);

                state(p) = STATE_LOC_VALID | STATE_COORD_VALID;

                ceres::Problem problem;

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
                    //FIXME this does not check geometry, only if its also on the surfaces (which might be good enough...)
                    auto ptr = test_surf->pointer();
                    if (test_surf->pointTo(ptr, coord, same_surface_th, 10,
                                                       surface_patch_index_ptr) <= same_surface_th) {
                        int count = 0;
                        int straight_count = 0;
                        state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
                        cv::Vec3f loc = test_surf->loc_raw(ptr);
                        data_th.loc(test_surf, p) = {loc[1], loc[0]};
                        float cost = local_cost(test_surf, p, data_th, state, points, step, src_step, &count, &straight_count, &coord);
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

            if (best_inliers >= curr_best_inl_th || best_ref_seed)
            {
                if (enforce_z_range && (best_coord[2] < z_min || best_coord[2] > z_max)) {
                    // Final guard: reject best candidate outside z-range
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
                        if (!state_sum && debug_images)
                            std::cout << "candidate rejected: nearest same-surface location had no valid state" << std::endl;
                    }
                }
                }
            }

            if (best_inliers >= curr_best_inl_th || best_ref_seed) {
                if (best_coord[0] == -1)
                    throw std::runtime_error("oops best_cord[0]");
                if (rollout_mode_active && !completes_existing_l_shape(p))
                    continue;

                data_th.surfs(p).insert(best_surf);
                data_th.loc(best_surf, p) = best_loc;
                state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
                generations(p) = static_cast<uint16_t>(std::min(generation + 1, static_cast<int>(std::numeric_limits<uint16_t>::max())));
                points(p) = best_coord;
                inliers_sum_dbg(p) = best_inliers;

                ceres::Problem problem;
                surftrack_add_local(best_surf, p, data_th, problem, state, points, step, src_step, SURF_LOSS | LOSS_ZLOC);

                std::vector<SurfacePatchIndex::LookupResult> more_local_hits;

                for(auto test_surf : local_surfs) {
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

                for (auto hit : surface_patch_hits(surface_patch_index_ptr, best_coord, best_surf)) {
                    QuadSurface* s = hit.surface.get();
                    if (s && !local_surfs.contains(s)) {
                        more_local_hits.push_back(std::move(hit));
                    }
                }

                ceres::Solver::Summary summary;

                ceres::Solve(options, &problem, &summary);

                //TODO only add/test if we have 2 neighs which both find locations
                for(const auto& hit : more_local_hits) {
                    QuadSurface* test_surf = hit.surface.get();
                    if (!test_surf) {
                        continue;
                    }
                    cv::Vec3f loc = test_surf->loc_raw(hit.ptr);
                    cv::Vec3f coord = SurfTrackerData::lookup_int_loc(test_surf, {loc[1], loc[0]});
                    if (coord[0] == -1) {
                        continue;
                    }
                    int count = 0;
                    float cost = local_cost_destructive(test_surf, p, data_th, state, points, step, src_step, loc, &count);
                    if (cost < local_cost_inl_th) {
                        data_th.loc(test_surf, p) = {loc[1], loc[0]};
                        data_th.surfs(p).insert(test_surf);
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

        if (succ > succ_before_generation) {
            emergency_consensus_limit_th = configured_consensus_limit_th;
        }

        int inl_lower_bound_reg = params.value("consensus_default_th", 10);
        int inl_lower_bound_b = emergency_consensus_limit_th;
        int inl_lower_bound = inl_lower_bound_reg;

        if (!at_right_border && curr_best_inl_th <= inl_lower_bound)
            inl_lower_bound = inl_lower_bound_b;

        const bool allow_threshold_retry = fringe.empty();
        if (allow_threshold_retry && curr_best_inl_th > inl_lower_bound) {
            const int smooth_next = std::max(inl_lower_bound, curr_best_inl_th - inlier_threshold_drop_step);
            const int best_failed_target = std::max(best_inliers_gen, inl_lower_bound);
            curr_best_inl_th = best_failed_target < curr_best_inl_th
                ? std::max(smooth_next, best_failed_target)
                : smooth_next;
            if (curr_best_inl_th >= inl_lower_bound) {
                cv::Rect active = active_bounds & used_area;
                reseed_fringe_from_valid(active);
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
        if (suppress_empty_fringe_reopt_after_component_drop &&
            loc_valid_count > component_drop_valid_count) {
            suppress_empty_fringe_reopt_after_component_drop = false;
            component_drop_reopt_skip_logged = false;
        }

        const bool reopt_interval_due = reopt_interval > 0 && generation > 0 && (generation % reopt_interval) == 0;
        bool update_mapping = reopt_interval > 0
            ? reopt_interval_due
            : (succ >= 1000 && (loc_valid_count-last_succ_parametrization) >= std::max(100.0, 0.3*last_succ_parametrization));
        if (fringe.empty() && suppress_empty_fringe_reopt_after_component_drop) {
            if (!component_drop_reopt_skip_logged) {
                std::cout << "optimizer: skipping empty-fringe reopt after component cleanup"
                          << " until new growth is accepted" << std::endl;
                component_drop_reopt_skip_logged = true;
            }
            update_mapping = false;
        } else if (fringe.empty() && final_opts && (reopt_interval == 0 || reopt_interval_due)) {
            final_opts--;
            update_mapping = true;
        }

        if (!global_steps_per_window && reopt_interval == 0)
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
                                         opt_data.seed_loc, closing_r, true, tgt_dir, surface_patch_index_ptr, debug_images,
                                         reopt_support_radius, reopt_robust_affine_support,
                                         reopt_drop_unsupported_points);
                const int valid_after_opt = count_loc_valid_in(opt_state, active);
                if (valid_before_opt > 0 && valid_after_opt * 2 < valid_before_opt) {
                    std::cout << "optimizer: rejecting mapping; valid points collapsed "
                              << valid_before_opt << " -> " << valid_after_opt << std::endl;
                } else {
                    copy(opt_data, data, active);
                    opt_points(active).copyTo(points(active));
                    opt_state(active).copyTo(state(active));
                    if (reopt_drop_disconnected_components &&
                        remove_components_disconnected_from_seed(data, state, points) > 0) {
                        loc_valid_count = count_loc_valid_in(state, used_area);
                        suppress_empty_fringe_reopt_after_component_drop = true;
                        component_drop_valid_count = loc_valid_count;
                        component_drop_reopt_skip_logged = false;
                    }
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
                reseed_fringe_from_valid(active);
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

        if (sweep_prune_distance_active &&
            generation >= sweep_prune_min_generations &&
            (generation % sweep_prune_interval_generations) == 0 &&
            loc_valid_count >= sweep_prune_min_valid_cells) {
            const int sample_stride = std::max(
                1,
                static_cast<int>(std::ceil(std::sqrt(
                    static_cast<double>(loc_valid_count) /
                    static_cast<double>(sweep_prune_max_samples)))));
            double sum_distance = 0.0;
            double max_distance = 0.0;
            int samples = 0;
            int misses = 0;
            const cv::Rect safe_used_area = used_area & cv::Rect(0, 0, state.cols, state.rows);
            for (int j = safe_used_area.y; j < safe_used_area.br().y; ++j) {
                for (int i = safe_used_area.x; i < safe_used_area.br().x; ++i) {
                    if ((j % sample_stride) != 0 || (i % sample_stride) != 0) {
                        continue;
                    }
                    if ((state(j, i) & STATE_LOC_VALID) == 0) {
                        continue;
                    }
                    const cv::Vec3d& p = points(j, i);
                    if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]) ||
                        p[0] == -1.0) {
                        continue;
                    }

                    SurfacePatchIndex::PointQuery query;
                    query.worldPoint = cv::Vec3f(
                        static_cast<float>(p[0]),
                        static_cast<float>(p[1]),
                        static_cast<float>(p[2]));
                    query.tolerance = static_cast<float>(sweep_prune_search_radius_vx);
                    auto hit = sweep_prune_target_index.locate(query);
                    if (!hit) {
                        misses++;
                        continue;
                    }
                    sum_distance += hit->distance;
                    max_distance = std::max(max_distance, static_cast<double>(hit->distance));
                    samples++;
                }
            }

            if (samples >= std::min(25, sweep_prune_min_valid_cells)) {
                const double mean_distance = sum_distance / static_cast<double>(samples);
                const double live_coverage =
                    std::clamp(static_cast<double>(current_area_vx2) / sweep_prune_target_area_vx2, 0.0, 1.0);
                const double coverage_adjusted_distance =
                    mean_distance / std::max(0.05, live_coverage);
                const double cutoff = std::max(
                    sweep_prune_best_coverage_adjusted_distance_vx * sweep_prune_distance_margin,
                    sweep_prune_best_coverage_adjusted_distance_vx + 1.0);
                std::cout << "sweep distance prune check: gen=" << generation
                          << " mean_vx=" << mean_distance
                          << " live_coverage=" << live_coverage
                          << " coverage_adjusted_vx=" << coverage_adjusted_distance
                          << " best_coverage_adjusted_vx=" << sweep_prune_best_coverage_adjusted_distance_vx
                          << " cutoff_vx=" << cutoff
                          << " samples=" << samples
                          << " misses=" << misses
                          << " max_vx=" << max_distance << std::endl;
                if (coverage_adjusted_distance > cutoff) {
                    std::cout << "sweep distance prune: stopping run at generation " << generation
                              << " because coverage-adjusted valid-cell mesh distance " << coverage_adjusted_distance
                              << " exceeds cutoff " << cutoff << std::endl;
                    break;
                }
            }
        }

        //continue expansion
        const int max_grid_extent = std::max(1, static_cast<int>(max_width / step));
        const int max_grid_height = std::max(1, static_cast<int>(max_height / step));
        const int max_grid_w = std::min(max_grid_extent, grid_limit_w);
        const int max_grid_h = std::min(max_grid_height, grid_limit_h);
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
        const bool can_expand_left = !growth_config.disable_grid_expansion && growth_config.grow_left && remaining_left > 0 && w < max_grid_w;
        const bool can_expand_right = !growth_config.disable_grid_expansion && growth_config.grow_right && remaining_right > 0 && w < max_grid_w;
        const bool can_expand_up = !growth_config.disable_grid_expansion && growth_config.grow_up && remaining_up > 0 && h < max_grid_h;
        const bool can_expand_down = !growth_config.disable_grid_expansion && growth_config.grow_down && remaining_down > 0 && h < max_grid_h;
        if (fringe.empty() && (can_expand_left || can_expand_right || can_expand_up || can_expand_down))
        {
            if (loc_valid_count <= last_expansion_loc_valid_count) {
                no_growth_expansions++;
            } else {
                no_growth_expansions = 0;
            }
            last_expansion_loc_valid_count = loc_valid_count;
            if (growth_config.max_no_growth_expansions > 0 && no_growth_expansions >= growth_config.max_no_growth_expansions) {
                std::cout << "stopping growth after " << no_growth_expansions
                          << " expansions with no valid-count increase"
                          << " (valid=" << loc_valid_count << ")" << std::endl;
                break;
            }
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
                reseed_fringe_from_valid(active);
            }
        }

        cv::imwrite((tgt_dir / "inliers_sum.tif").string(), inliers_sum_dbg(used_area));

        const int current_remaining_left = remaining_extra(max_extra_left, expanded_left);
        const int current_remaining_right = remaining_extra(max_extra_right, expanded_right);
        const bool can_expand_width_now =
            !growth_config.disable_grid_expansion &&
            ((growth_config.grow_left && current_remaining_left > 0 && w < max_grid_w) ||
             (growth_config.grow_right && current_remaining_right > 0 && w < max_grid_w));
        const bool max_width_limit_reached =
            max_grid_extent <= grid_limit_w && w >= max_grid_w;
        const bool horizontal_expansion_blocked_by_max_width =
            !growth_config.disable_grid_expansion &&
            max_width_limit_reached &&
            (growth_config.grow_left || growth_config.grow_right);

        if (configured_consensus_limit_th > 2 &&
            fringe.empty() &&
            emergency_consensus_limit_th > 2 &&
            !can_expand_width_now &&
            horizontal_expansion_blocked_by_max_width) {
            --emergency_consensus_limit_th;
            curr_best_inl_th = emergency_consensus_limit_th;
            cv::Rect active = active_bounds & used_area;
            reseed_fringe_from_valid(active);
            if (fringe.empty()) {
                reseed_fringe_from_valid(used_area);
            }
            if (!fringe.empty()) {
                std::cout << "last-chance consensus retry at inl_th "
                          << curr_best_inl_th
                          << " after stalled expansion" << std::endl;
                continue;
            }
            std::cout << "last-chance consensus retry at inl_th "
                      << curr_best_inl_th
                      << " could not reseed after width-limited stalled expansion" << std::endl;
        }

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
