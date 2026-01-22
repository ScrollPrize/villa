#include "vc/core/util/SurfaceHelpers.hpp"

#include <omp.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace vc::surface_helpers {

// Global parameters
static LossParams g_params;

void set_loss_params(const LossParams& params) {
    g_params = params;
}

const LossParams& get_loss_params() {
    return g_params;
}

// ============================================================================
// Loss Function Implementations
// ============================================================================

int add_surftrack_distloss(
    QuadSurface* sm,
    const cv::Vec2i& p,
    const cv::Vec2i& off,
    SurfTrackerData& data,
    ceres::Problem& problem,
    const cv::Mat_<uint8_t>& state,
    float unit,
    int flags,
    ceres::ResidualBlockId* res,
    float w)
{
    if (!point_in_bounds(state, p) || !point_in_bounds(state, p + off))
        return 0;
    if ((state(p) & STATE_LOC_VALID) == 0 || !data.has(sm, p))
        return 0;
    if ((state(p+off) & STATE_LOC_VALID) == 0 || !data.has(sm, p+off))
        return 0;

    // Use the global parameter if w is default value (1.0), otherwise use the provided value
    float weight = (w == 1.0f) ? g_params.dist_loss_2d_w : w;
    ceres::ResidualBlockId tmp = problem.AddResidualBlock(
        DistLoss2D::Create(unit*cv::norm(off), weight),
        nullptr,
        &data.loc(sm, p)[0],
        &data.loc(sm, p+off)[0]);
    if (res)
        *res = tmp;

    if ((flags & OPTIMIZE_ALL) == 0)
        problem.SetParameterBlockConstant(&data.loc(sm, p+off)[0]);

    return 1;
}

int add_surftrack_distloss_3D(
    cv::Mat_<cv::Vec3d>& points,
    const cv::Vec2i& p,
    const cv::Vec2i& off,
    ceres::Problem& problem,
    const cv::Mat_<uint8_t>& state,
    float unit,
    int flags,
    ceres::ResidualBlockId* res,
    float w)
{
    if (!point_in_bounds(state, p) || !point_in_bounds(state, p + off))
        return 0;
    if ((state(p) & (STATE_COORD_VALID | STATE_LOC_VALID)) == 0)
        return 0;
    if ((state(p+off) & (STATE_COORD_VALID | STATE_LOC_VALID)) == 0)
        return 0;

    // Use the global parameter if w is default value (2.0), otherwise use the provided value
    float weight = (w == 2.0f) ? g_params.dist_loss_3d_w : w;
    ceres::ResidualBlockId tmp = problem.AddResidualBlock(
        DistLoss::Create(unit*cv::norm(off), weight),
        nullptr,
        &points(p)[0],
        &points(p+off)[0]);

    if (res)
        *res = tmp;

    if ((flags & OPTIMIZE_ALL) == 0)
        problem.SetParameterBlockConstant(&points(p+off)[0]);

    return 1;
}

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
    int flags)
{
    resId_t id(type, sm, p, p+off);
    if (data.hasResId(id))
        return 0;

    ceres::ResidualBlockId res;
    int count = add_surftrack_distloss_3D(points, p, off, problem, state, unit, flags, &res);

    data.resId(id) = res;

    return count;
}

int cond_surftrack_distloss(
    int type,
    QuadSurface* sm,
    const cv::Vec2i& p,
    const cv::Vec2i& off,
    SurfTrackerData& data,
    ceres::Problem& problem,
    const cv::Mat_<uint8_t>& state,
    float unit,
    int flags)
{
    resId_t id(type, sm, p, p+off);
    if (data.hasResId(id))
        return 0;

    add_surftrack_distloss(sm, p, off, data, problem, state, unit, flags, &data.resId(id));

    return 1;
}

int add_surftrack_straightloss(
    QuadSurface* sm,
    const cv::Vec2i& p,
    const cv::Vec2i& o1,
    const cv::Vec2i& o2,
    const cv::Vec2i& o3,
    SurfTrackerData& data,
    ceres::Problem& problem,
    const cv::Mat_<uint8_t>& state,
    int flags,
    float w)
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
    w = g_params.straight_weight;

    problem.AddResidualBlock(
        StraightLoss2D::Create(w),
        nullptr,
        &data.loc(sm, p+o1)[0],
        &data.loc(sm, p+o2)[0],
        &data.loc(sm, p+o3)[0]);

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

int add_surftrack_straightloss_3D(
    const cv::Vec2i& p,
    const cv::Vec2i& o1,
    const cv::Vec2i& o2,
    const cv::Vec2i& o3,
    cv::Mat_<cv::Vec3d>& points,
    ceres::Problem& problem,
    const cv::Mat_<uint8_t>& state,
    int flags,
    ceres::ResidualBlockId* res,
    float w)
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
    w = g_params.straight_weight_3D;

    ceres::ResidualBlockId tmp = problem.AddResidualBlock(
        StraightLoss::Create(w),
        nullptr,
        &points(p+o1)[0],
        &points(p+o2)[0],
        &points(p+o3)[0]);
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
    int flags)
{
    resId_t id(type, sm, p);
    if (data.hasResId(id))
        return 0;

    ceres::ResidualBlockId res;
    int count = add_surftrack_straightloss_3D(p, o1, o2, o3, points, problem, state, flags, &res);

    if (count)
        data.resId(id) = res;

    return count;
}

int add_surftrack_surfloss(
    QuadSurface* sm,
    const cv::Vec2i& p,
    SurfTrackerData& data,
    ceres::Problem& problem,
    const cv::Mat_<uint8_t>& state,
    cv::Mat_<cv::Vec3d>& points,
    float step,
    ceres::ResidualBlockId* res,
    float w)
{
    if ((state(p) & STATE_LOC_VALID) == 0 || !data.valid_int(sm, p))
        return 0;

    ceres::ResidualBlockId tmp = problem.AddResidualBlock(
        SurfaceLossD::Create(sm->rawPoints(), w),
        nullptr,
        &points(p)[0],
        &data.loc(sm, p)[0]);

    if (res)
        *res = tmp;

    return 1;
}

int cond_surftrack_surfloss(
    int type,
    QuadSurface* sm,
    const cv::Vec2i& p,
    SurfTrackerData& data,
    ceres::Problem& problem,
    const cv::Mat_<uint8_t>& state,
    cv::Mat_<cv::Vec3d>& points,
    float step)
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

int surftrack_add_local(
    QuadSurface* sm,
    const cv::Vec2i& p,
    SurfTrackerData& data,
    ceres::Problem& problem,
    const cv::Mat_<uint8_t>& state,
    cv::Mat_<cv::Vec3d>& points,
    float step,
    float src_step,
    int flags,
    int* straigh_count_ptr)
{
    int count = 0;
    int count_straight = 0;

    // Always add 2D surface parameterization losses
    count += add_surftrack_distloss(sm, p, {0,1}, data, problem, state, step);
    count += add_surftrack_distloss(sm, p, {1,0}, data, problem, state, step);
    count += add_surftrack_distloss(sm, p, {0,-1}, data, problem, state, step);
    count += add_surftrack_distloss(sm, p, {-1,0}, data, problem, state, step);

    // diagonal
    count += add_surftrack_distloss(sm, p, {1,1}, data, problem, state, step);
    count += add_surftrack_distloss(sm, p, {1,-1}, data, problem, state, step);
    count += add_surftrack_distloss(sm, p, {-1,1}, data, problem, state, step);
    count += add_surftrack_distloss(sm, p, {-1,-1}, data, problem, state, step);

    // horizontal 2D straightness
    count_straight += add_surftrack_straightloss(sm, p, {0,-2},{0,-1},{0,0}, data, problem, state);
    count_straight += add_surftrack_straightloss(sm, p, {0,-1},{0,0},{0,1}, data, problem, state);
    count_straight += add_surftrack_straightloss(sm, p, {0,0},{0,1},{0,2}, data, problem, state);

    // vertical 2D straightness
    count_straight += add_surftrack_straightloss(sm, p, {-2,0},{-1,0},{0,0}, data, problem, state);
    count_straight += add_surftrack_straightloss(sm, p, {-1,0},{0,0},{1,0}, data, problem, state);
    count_straight += add_surftrack_straightloss(sm, p, {0,0},{1,0},{2,0}, data, problem, state);

    // Additionally add 3D losses when LOSS_3D_INDIRECT is set
    if (flags & LOSS_3D_INDIRECT) {
        // 3D distance losses
        count += add_surftrack_distloss_3D(points, p, {0,1}, problem, state, step*src_step, flags);
        count += add_surftrack_distloss_3D(points, p, {1,0}, problem, state, step*src_step, flags);
        count += add_surftrack_distloss_3D(points, p, {0,-1}, problem, state, step*src_step, flags);
        count += add_surftrack_distloss_3D(points, p, {-1,0}, problem, state, step*src_step, flags);

        // 3D diagonal distance losses
        count += add_surftrack_distloss_3D(points, p, {1,1}, problem, state, step*src_step, flags);
        count += add_surftrack_distloss_3D(points, p, {1,-1}, problem, state, step*src_step, flags);
        count += add_surftrack_distloss_3D(points, p, {-1,1}, problem, state, step*src_step, flags);
        count += add_surftrack_distloss_3D(points, p, {-1,-1}, problem, state, step*src_step, flags);

        // 3D straightness losses
        count_straight += add_surftrack_straightloss_3D(p, {0,-2},{0,-1},{0,0}, points, problem, state);
        count_straight += add_surftrack_straightloss_3D(p, {0,-1},{0,0},{0,1}, points, problem, state);
        count_straight += add_surftrack_straightloss_3D(p, {0,0},{0,1},{0,2}, points, problem, state);

        count_straight += add_surftrack_straightloss_3D(p, {-2,0},{-1,0},{0,0}, points, problem, state);
        count_straight += add_surftrack_straightloss_3D(p, {-1,0},{0,0},{1,0}, points, problem, state);
        count_straight += add_surftrack_straightloss_3D(p, {0,0},{1,0},{2,0}, points, problem, state);
    }

    if (flags & LOSS_ZLOC)
        problem.AddResidualBlock(
            ZLocationLoss<cv::Vec3f>::Create(
                sm->rawPoints(),
                data.seed_coord[2] - (p[0]-data.seed_loc[0])*step*src_step, g_params.z_loc_loss_w),
            new ceres::HuberLoss(1.0),
            &data.loc(sm, p)[0]);

    if (flags & SURF_LOSS) {
        count += add_surftrack_surfloss(sm, p, data, problem, state, points, step);
    }

    if (straigh_count_ptr)
        *straigh_count_ptr += count_straight;

    return count + count_straight;
}

int surftrack_add_global(
    QuadSurface* sm,
    const cv::Vec2i& p,
    SurfTrackerData& data,
    ceres::Problem& problem,
    const cv::Mat_<uint8_t>& state,
    cv::Mat_<cv::Vec3d>& points,
    float step,
    int flags,
    float step_onsurf)
{
    if ((state(p) & (STATE_LOC_VALID | STATE_COORD_VALID)) == 0)
        return 0;

    int count = 0;
    // losses are defined in 3D
    if (flags & LOSS_3D_INDIRECT) {
        // h
        count += cond_surftrack_distloss_3D(0, sm, points, p, {0,1}, data, problem, state, step, flags);
        count += cond_surftrack_distloss_3D(0, sm, points, p, {1,0}, data, problem, state, step, flags);
        count += cond_surftrack_distloss_3D(1, sm, points, p, {0,-1}, data, problem, state, step, flags);
        count += cond_surftrack_distloss_3D(1, sm, points, p, {-1,0}, data, problem, state, step, flags);

        // v
        count += cond_surftrack_distloss_3D(2, sm, points, p, {1,1}, data, problem, state, step, flags);
        count += cond_surftrack_distloss_3D(2, sm, points, p, {1,-1}, data, problem, state, step, flags);
        count += cond_surftrack_distloss_3D(3, sm, points, p, {-1,1}, data, problem, state, step, flags);
        count += cond_surftrack_distloss_3D(3, sm, points, p, {-1,-1}, data, problem, state, step, flags);

        // horizontal
        count += cond_surftrack_straightloss_3D(4, sm, p, {0,-2},{0,-1},{0,0}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(4, sm, p, {0,-1},{0,0},{0,1}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(4, sm, p, {0,0},{0,1},{0,2}, points, data, problem, state, flags);

        // vertical
        count += cond_surftrack_straightloss_3D(5, sm, p, {-2,0},{-1,0},{0,0}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(5, sm, p, {-1,0},{0,0},{1,0}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(5, sm, p, {0,0},{1,0},{2,0}, points, data, problem, state, flags);

        // dia1
        count += cond_surftrack_straightloss_3D(6, sm, p, {-2,-2},{-1,-1},{0,0}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(6, sm, p, {-1,-1},{0,0},{1,1}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(6, sm, p, {0,0},{1,1},{2,2}, points, data, problem, state, flags);

        // dia2
        count += cond_surftrack_straightloss_3D(7, sm, p, {-2,2},{-1,1},{0,0}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(7, sm, p, {-1,1},{0,0},{1,-1}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(7, sm, p, {0,0},{1,-1},{2,-2}, points, data, problem, state, flags);
    }

    // losses on surface
    if (flags & LOSS_ON_SURF) {
        if (step_onsurf == 0)
            throw std::runtime_error("oops step_onsurf == 0");

        // direct
        count += cond_surftrack_distloss(8, sm, p, {0,1}, data, problem, state, step_onsurf);
        count += cond_surftrack_distloss(8, sm, p, {1,0}, data, problem, state, step_onsurf);
        count += cond_surftrack_distloss(9, sm, p, {0,-1}, data, problem, state, step_onsurf);
        count += cond_surftrack_distloss(9, sm, p, {-1,0}, data, problem, state, step_onsurf);

        // diagonal
        count += cond_surftrack_distloss(10, sm, p, {1,1}, data, problem, state, step_onsurf);
        count += cond_surftrack_distloss(10, sm, p, {1,-1}, data, problem, state, step_onsurf);
        count += cond_surftrack_distloss(11, sm, p, {-1,1}, data, problem, state, step_onsurf);
        count += cond_surftrack_distloss(11, sm, p, {-1,-1}, data, problem, state, step_onsurf);
    }

    if ((flags & SURF_LOSS) && (state(p) & STATE_LOC_VALID))
        count += cond_surftrack_surfloss(14, sm, p, data, problem, state, points, step);

    return count;
}

// ============================================================================
// Local Cost Evaluation
// ============================================================================

int local_cost_patch_radius() {
    return 2;
}

double local_cost_destructive(
    QuadSurface* sm,
    const cv::Vec2i& p,
    SurfTrackerData& data,
    cv::Mat_<uint8_t>& state,
    cv::Mat_<cv::Vec3d>& points,
    float step,
    float src_step,
    cv::Vec3f loc,
    int* ref_count,
    int* straight_count_ptr)
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
    }
    data.erase(sm, p);
    state(p) = state_old;

    if (!count)
        return 0;
    else
        return sqrt(test_loss/count);
}

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
    int* ref_count,
    int* straight_count_ptr)
{
    struct LocalCostScratch {
        int size = 0;
        cv::Mat_<uint8_t> state;
        cv::Mat_<cv::Vec3d> points;
        SurfTrackerData data;
    };
    static thread_local LocalCostScratch scratch;

    const int radius = local_cost_patch_radius();
    const int size = radius * 2 + 1;
    if (scratch.size != size) {
        scratch.size = size;
        scratch.state.create(size, size);
        scratch.points.create(size, size);
    }
    scratch.state.setTo(0);
    scratch.points.setTo(cv::Scalar(-1, -1, -1));
    scratch.data.clear();
    cv::Mat_<uint8_t>& state_local = scratch.state;
    cv::Mat_<cv::Vec3d>& points_local = scratch.points;
    SurfTrackerData& data_local = scratch.data;

    const cv::Vec2i origin = {p[0] - radius, p[1] - radius};
    const cv::Rect bounds(0, 0, state_src.cols, state_src.rows);

    {
        std::shared_lock<std::shared_mutex> lock(mutex);
        for (int y = 0; y < size; ++y) {
            const int gy = origin[0] + y;
            if (gy < 0 || gy >= bounds.height)
                continue;
            for (int x = 0; x < size; ++x) {
                const int gx = origin[1] + x;
                if (gx < 0 || gx >= bounds.width)
                    continue;
                state_local(y, x) = state_src(gy, gx);
                points_local(y, x) = points_src(gy, gx);
                cv::Vec2d loc_tmp;
                if (data_src.getLoc(sm, {gy, gx}, &loc_tmp))
                    data_local.loc(sm, {y, x}) = loc_tmp;
            }
        }
    }

    const cv::Vec2i p_local = {radius, radius};
    state_local(p_local) = STATE_LOC_VALID | STATE_COORD_VALID;
    data_local.loc(sm, p_local) = {static_cast<double>(loc[1]), static_cast<double>(loc[0])};

    return local_cost(sm, p_local, data_local, state_local, points_local, step, src_step,
                      ref_count, straight_count_ptr);
}

double local_cost(
    QuadSurface* sm,
    const cv::Vec2i& p,
    SurfTrackerData& data,
    const cv::Mat_<uint8_t>& state,
    cv::Mat_<cv::Vec3d>& points,
    float step,
    float src_step,
    int* ref_count,
    int* straight_count_ptr)
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

double local_solve(
    QuadSurface* sm,
    const cv::Vec2i& p,
    SurfTrackerData& data,
    const cv::Mat_<uint8_t>& state,
    cv::Mat_<cv::Vec3d>& points,
    float step,
    float src_step,
    int flags)
{
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

    return summary.final_cost / summary.num_residual_blocks;
}

// ============================================================================
// Generation/HR Output
// ============================================================================

int surftrack_round_step(float step) {
    return std::max(1, static_cast<int>(std::lround(step)));
}

cv::Mat_<uint16_t> surftrack_generation_channel(
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

uint16_t surftrack_max_generation(
    const cv::Mat_<uint16_t>& generations,
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

cv::Mat_<cv::Vec3f> surftrack_genpoints_hr(
    SurfTrackerData& data,
    cv::Mat_<uint8_t>& state,
    cv::Mat_<cv::Vec3d>& points,
    const cv::Rect& used_area,
    float step,
    float step_src,
    bool inpaint,
    bool parallel)
{
    std::cout << "hr_gen: start used_area=" << used_area << " step=" << step
              << " inpaint=" << inpaint << " parallel=" << (parallel?1:0) << std::endl;
    cv::Mat_<cv::Vec3f> points_hr(state.rows*step, state.cols*step, {0,0,0});
    cv::Mat_<float> weights_hr(state.rows*step, state.cols*step, 0.0f);
#pragma omp parallel for if(parallel)
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
            // Inpaint each missing HR sample individually
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

void thin_plate_inpaint(
    cv::Mat_<cv::Vec3d>& points,
    const cv::Mat_<uint8_t>& state,
    const cv::Mat_<uint8_t>& inpaint_mask,
    const cv::Rect& used_area,
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
        if ((state(yy, xx) & STATE_VALID) == 0 && !inpaint_mask(yy, xx))
            return fallback;
        return points(yy, xx);
    };

    // Warmup: diffuse boundary values into holes
    cv::Mat_<uint8_t> warm_valid;
    cv::bitwise_and(state, STATE_VALID, warm_valid);
    warm_valid.setTo(0, inpaint_mask);
    for (int iter = 0; iter < warmup_iters; ++iter) {
        for (int y = y0; y < y1; ++y)
            for (int x = x0; x < x1; ++x)
                if (inpaint_mask(y, x)) {
                    cv::Vec3d sum = {0, 0, 0};
                    int count = 0;

                    auto add_if_valid = [&](int yy, int xx) {
                        yy = clamp_i(yy, 0, max_y);
                        xx = clamp_i(xx, 0, max_x);
                        if (warm_valid(yy, xx)) {
                            sum += points(yy, xx);
                            count++;
                        }
                    };

                    add_if_valid(y - 1, x);
                    add_if_valid(y + 1, x);
                    add_if_valid(y, x - 1);
                    add_if_valid(y, x + 1);

                    if (count > 0) {
                        points(y, x) = sum / count;
                        warm_valid(y, x) = 1;
                    }
                }
    }

    // Thin-plate (bi-Laplacian) smoothing using a 13-point stencil
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

}  // namespace vc::surface_helpers
