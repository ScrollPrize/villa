#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/SurfaceModeling.hpp"
#include "vc/core/util/OMPThreadPointCollection.hpp"
#include "vc/core/util/LifeTime.hpp"
#include "vc/core/types/ChunkedTensor.hpp"
#include <nlohmann/json.hpp>


#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(views, xview.hpp)

#include <iostream>

#include "vc/tracer/Tracer.hpp"

static float space_trace_dist_w = 1.0;
float dist_th = 1.5;

// global CUDA to allow use to set to false globally
// in the case they have cuda avail, but do not want to use it
static bool g_use_cuda = true;

// Expose a simple toggle for CUDA usage so tools can honor JSON settings
void set_space_tracing_use_cuda(bool enable) {
    g_use_cuda = enable;
}


static int gen_straight_loss(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &o1, const cv::Vec2i &o2, const cv::Vec2i &o3, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &dpoints, bool optimize_all, float w = 0.5);
static int gen_dist_loss(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &dpoints, float unit, bool optimize_all, ceres::ResidualBlockId *res, float w = 1.0);

static bool loc_valid(int state)
{
    return state & STATE_LOC_VALID;
}

static bool coord_valid(int state)
{
    return (state & STATE_COORD_VALID) || (state & STATE_LOC_VALID);
}

//gen straigt loss given point and 3 offsets
static int gen_straight_loss(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &o1, const cv::Vec2i &o2,
    const cv::Vec2i &o3, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &dpoints, bool optimize_all, float w)
{
    if (!coord_valid(state(p+o1)))
        return 0;
    if (!coord_valid(state(p+o2)))
        return 0;
    if (!coord_valid(state(p+o3)))
        return 0;

    problem.AddResidualBlock(StraightLoss::Create(w), nullptr, &dpoints(p+o1)[0], &dpoints(p+o2)[0], &dpoints(p+o3)[0]);

    if (!optimize_all) {
        if (o1 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&dpoints(p+o1)[0]);
        if (o2 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&dpoints(p+o2)[0]);
        if (o3 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&dpoints(p+o3)[0]);
    }

    return 1;
}

static int gen_dist_loss(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &dpoints,
    float unit, bool optimize_all, ceres::ResidualBlockId *res, float w)
{
    // Add a loss saying that dpoints(p) and dpoints(p+off) should themselves be distance |off| apart
    // Here dpoints is a 2D grid mapping surface-space points to 3D volume space
    // So this says that distances should be preserved from volume to surface

    if (!coord_valid(state(p)))
        return 0;
    if (!coord_valid(state(p+off)))
        return 0;

    ceres::ResidualBlockId tmp = problem.AddResidualBlock(DistLoss::Create(unit*cv::norm(off),w), nullptr, &dpoints(p)[0], &dpoints(p+off)[0]);

    if (res)
        *res = tmp;

    if (!optimize_all)
        problem.SetParameterBlockConstant(&dpoints(p+off)[0]);

    return 1;
}

static cv::Vec2i lower_p(const cv::Vec2i &point, const cv::Vec2i &offset)
{
    if (offset[0] == 0) {
        if (offset[1] < 0)
            return point+offset;
        else
            return point;
    }
    if (offset[0] < 0)
        return point+offset;
    else
        return point;
}

static bool loss_mask(int bit, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint16_t> &loss_status)
{
    return loss_status(lower_p(p, off)) & (1 << bit);
}

static int set_loss_mask(int bit, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint16_t> &loss_status, int set)
{
    if (set)
        loss_status(lower_p(p, off)) |= (1 << bit);
    return set;
}

static int conditional_dist_loss(int bit, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint16_t> &loss_status,
    ceres::Problem &problem, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &out, float unit, bool optimize_all, float w = 1.0)
{
    int set = 0;
    if (!loss_mask(bit, p, off, loss_status))
        set = set_loss_mask(bit, p, off, loss_status, gen_dist_loss(problem, p, off, state, out, unit, optimize_all, nullptr, w));
    return set;
};

static int conditional_straight_loss(int bit, const cv::Vec2i &p, const cv::Vec2i &o1, const cv::Vec2i &o2, const cv::Vec2i &o3,
    cv::Mat_<uint16_t> &loss_status, ceres::Problem &problem, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &out, bool optimize_all)
{
    int set = 0;
    if (!loss_mask(bit, p, o2, loss_status))
        set += set_loss_mask(bit, p, o2, loss_status, gen_straight_loss(problem, p, o1, o2, o3, state, out, optimize_all));
    return set;
};



static void freeze_inner_params(ceres::Problem &problem, int edge_dist, const cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &out,
    cv::Mat_<cv::Vec2d> &loc, cv::Mat_<uint16_t> &loss_status, int inner_flags)
{
    cv::Mat_<float> dist(state.size());

    edge_dist = std::min(edge_dist,254);


    cv::Mat_<uint8_t> masked;
    bitwise_and(state, cv::Scalar(inner_flags), masked);


    cv::distanceTransform(masked, dist, cv::DIST_L1, cv::DIST_MASK_3);

    for(int j=0;j<dist.rows;j++)
        for(int i=0;i<dist.cols;i++) {
            if (dist(j,i) >= edge_dist && !loss_mask(7, {j,i}, {0,0}, loss_status)) {
                if (problem.HasParameterBlock(&out(j,i)[0]))
                    problem.SetParameterBlockConstant(&out(j,i)[0]);
                if (!loc.empty() && problem.HasParameterBlock(&loc(j,i)[0]))
                    problem.SetParameterBlockConstant(&loc(j,i)[0]);
                set_loss_mask(7, {j,i}, {0,0}, loss_status, 1);
            }
            if (dist(j,i) >= edge_dist+1 && !loss_mask(8, {j,i}, {0,0}, loss_status)) {
                if (problem.HasParameterBlock(&out(j,i)[0]))
                    problem.RemoveParameterBlock(&out(j,i)[0]);
                if (!loc.empty() && problem.HasParameterBlock(&loc(j,i)[0]))
                    problem.RemoveParameterBlock(&loc(j,i)[0]);
                set_loss_mask(8, {j,i}, {0,0}, loss_status, 1);
            }
        }
}


struct DSReader
{
    z5::Dataset *ds;
    float scale;
    ChunkCache *cache;
};


template <typename T, typename C>
static int gen_space_loss(ceres::Problem &problem, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &loc, Chunked3d<T,C> &t, float w = 0.1)
{
    // Add a loss saying that value of 3D volume tensor t at location loc(p) should be near-zero

    if (!loc_valid(state(p)))
        return 0;

    problem.AddResidualBlock(SpaceLossAcc<T,C>::Create(t, w), nullptr, &loc(p)[0]);

    return 1;
}

template <typename T, typename C>
static int gen_space_line_loss(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint8_t> &state,
    cv::Mat_<cv::Vec3d> &loc, Chunked3d<T,C> &t, int steps, float w = 0.1, float dist_th = 2)
{
    // Add a loss saying that value of 3D volume tensor t should be near-zero for all locations along
    // the line from loc(p) to loc(p + off)

    if (!loc_valid(state(p)))
        return 0;
    if (!loc_valid(state(p+off)))
        return 0;

    problem.AddResidualBlock(SpaceLineLossAcc<T,C>::Create(t, steps, w), nullptr, &loc(p)[0], &loc(p+off)[0]);

    return 1;
}

int gen_direction_loss(ceres::Problem &problem, const cv::Vec2i &p, const int off_dist, cv::Mat_<uint8_t> &state,
    cv::Mat_<cv::Vec3d> &loc, std::vector<DirectionField> const &direction_fields, float w = 1.f)
{
    // Add losses saying that the local basis vectors of the patch at loc(p) should match those of the given fields

    if (!loc_valid(state(p)))
        return 0;

    cv::Vec2i const p_off_horz{p[0], p[1] + off_dist};
    cv::Vec2i const p_off_vert{p[0] + off_dist, p[1]};

    int count = 0;
    for (auto &field: direction_fields) {
        if (field.direction == "horizontal") {
            if (!loc_valid(state(p_off_horz)))
                continue;
            problem.AddResidualBlock(FiberDirectionLoss::Create(*field.field_ptr, w), nullptr, &loc(p)[0], &loc(p_off_horz)[0]);
        } else if (field.direction == "vertical") {
            if (!loc_valid(state(p_off_vert)))
                continue;
            problem.AddResidualBlock(FiberDirectionLoss::Create(*field.field_ptr, w), nullptr, &loc(p)[0], &loc(p_off_vert)[0]);
        } else if (field.direction == "normal") {
            if (!loc_valid(state(p_off_horz)) || !loc_valid(state(p_off_vert)))
                continue;
            problem.AddResidualBlock(NormalDirectionLoss::Create(*field.field_ptr, w), nullptr, &loc(p)[0], &loc(p_off_horz)[0], &loc(p_off_vert)[0]);
        } else {
            assert(false);
        }
        ++count;
    }

    return count;
}

//create all valid losses for this point
template <typename I, typename T, typename C>
static int emptytrace_create_centered_losses(ceres::Problem &problem, const cv::Vec2i &p, cv::Mat_<uint8_t> &state,
    cv::Mat_<cv::Vec3d> &loc, const I &interp, Chunked3d<T,C> &t, std::vector<DirectionField> const &direction_fields,
    float unit, int flags = 0)
{
    //generate losses for point p
    int count = 0;

    if (p[0] < 2 || p[1] < 2 || p[1] >= state.cols-2 || p[0] >= state.rows-2)
        throw std::runtime_error("point too close to problem border!");

    //horizontal
    count += gen_straight_loss(problem, p, {0,-2},{0,-1},{0,0}, state, loc, flags & OPTIMIZE_ALL);
    count += gen_straight_loss(problem, p, {0,-1},{0,0},{0,1}, state, loc, flags & OPTIMIZE_ALL);
    count += gen_straight_loss(problem, p, {0,0},{0,1},{0,2}, state, loc, flags & OPTIMIZE_ALL);

    //vertical
    count += gen_straight_loss(problem, p, {-2,0},{-1,0},{0,0}, state, loc, flags & OPTIMIZE_ALL);
    count += gen_straight_loss(problem, p, {-1,0},{0,0},{1,0}, state, loc, flags & OPTIMIZE_ALL);
    count += gen_straight_loss(problem, p, {0,0},{1,0},{2,0}, state, loc, flags & OPTIMIZE_ALL);

    //direct neighboars
    count += gen_dist_loss(problem, p, {0,-1}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);
    count += gen_dist_loss(problem, p, {0,1}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);
    count += gen_dist_loss(problem, p, {-1,0}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);
    count += gen_dist_loss(problem, p, {1,0}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);

    //diagonal neighbors
    count += gen_dist_loss(problem, p, {1,-1}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);
    count += gen_dist_loss(problem, p, {-1,1}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);
    count += gen_dist_loss(problem, p, {1,1}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);
    count += gen_dist_loss(problem, p, {-1,-1}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);

    if (flags & SPACE_LOSS) {
        count += gen_space_loss(problem, p, state, loc, t);

        count += gen_space_line_loss(problem, p, {1,0}, state, loc, t, unit);
        count += gen_space_line_loss(problem, p, {-1,0}, state, loc, t, unit);
        count += gen_space_line_loss(problem, p, {0,1}, state, loc, t, unit);
        count += gen_space_line_loss(problem, p, {0,-1}, state, loc, t, unit);

        count += gen_direction_loss(problem, p, 1, state, loc, direction_fields);
        count += gen_direction_loss(problem, p, -1, state, loc, direction_fields);
    }

    return count;
}

template <typename T, typename C>
static int conditional_spaceline_loss(int bit, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint16_t> &loss_status,
    ceres::Problem &problem, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &loc, Chunked3d<T,C> &t, int steps)
{
    int set = 0;
    if (!loss_mask(bit, p, off, loss_status))
        set = set_loss_mask(bit, p, off, loss_status, gen_space_line_loss(problem, p, off, state, loc, t, steps));
    return set;
};

static int conditional_direction_loss(int bit, const cv::Vec2i &p, const int u_off, cv::Mat_<uint16_t> &loss_status,
    ceres::Problem &problem, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &loc, std::vector<DirectionField> const &direction_fields)
{
    int set = 0;
    cv::Vec2i const off{0, u_off};
    if (!loss_mask(bit, p, off, loss_status))
        set = set_loss_mask(bit, p, off, loss_status, gen_direction_loss(problem, p, u_off, state, loc, direction_fields));
    return set;
};

//create only missing losses so we can optimize the whole problem
template <typename I, typename T, typename C>
static int emptytrace_create_missing_centered_losses(ceres::Problem &problem, cv::Mat_<uint16_t> &loss_status, const cv::Vec2i &p,
    cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &loc, const I &interp, Chunked3d<T,C> &t, std::vector<DirectionField> const &direction_fields,
    float unit, int flags = SPACE_LOSS | OPTIMIZE_ALL)
{
    //generate losses for point p
    int count = 0;

    //horizontal
    // if (flags & SPACE_LOSS) {
        count += conditional_straight_loss(0, p, {0,-2},{0,-1},{0,0}, loss_status, problem, state, loc, flags);
        count += conditional_straight_loss(0, p, {0,-1},{0,0},{0,1}, loss_status, problem, state, loc, flags);
        count += conditional_straight_loss(0, p, {0,0},{0,1},{0,2}, loss_status, problem, state, loc, flags);

        //vertical
        count += conditional_straight_loss(1, p, {-2,0},{-1,0},{0,0}, loss_status, problem, state, loc, flags);
        count += conditional_straight_loss(1, p, {-1,0},{0,0},{1,0}, loss_status, problem, state, loc, flags);
        count += conditional_straight_loss(1, p, {0,0},{1,0},{2,0}, loss_status, problem, state, loc, flags);
    // }

    //direct neighboars h
    count += conditional_dist_loss(2, p, {0,-1}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);
    count += conditional_dist_loss(2, p, {0,1}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);

    //direct neighbors v
    count += conditional_dist_loss(3, p, {-1,0}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);
    count += conditional_dist_loss(3, p, {1,0}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);

    //diagonal neighbors
    count += conditional_dist_loss(4, p, {1,-1}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);
    count += conditional_dist_loss(4, p, {-1,1}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);

    count += conditional_dist_loss(5, p, {1,1}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);
    count += conditional_dist_loss(5, p, {-1,-1}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);

    if (flags & SPACE_LOSS) {
        if (!loss_mask(6, p, {0,0}, loss_status))
            count += set_loss_mask(6, p, {0,0}, loss_status, gen_space_loss(problem, p, state, loc, t));

        count += conditional_spaceline_loss(7, p, {1,0}, loss_status, problem, state, loc, t, unit);
        count += conditional_spaceline_loss(7, p, {-1,0}, loss_status, problem, state, loc, t, unit);

        count += conditional_spaceline_loss(8, p, {0,1}, loss_status, problem, state, loc, t, unit);
        count += conditional_spaceline_loss(8, p, {0,-1}, loss_status, problem, state, loc, t, unit);

        count += conditional_direction_loss(9, p, 1, loss_status, problem, state, loc, direction_fields);
        count += conditional_direction_loss(9, p, -1, loss_status, problem, state, loc, direction_fields);
    }

    return count;
}

//optimize within a radius, setting edge points to constant
template <typename I, typename T, typename C>
static float local_optimization(int radius, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &locs,
    const I &interp, Chunked3d<T,C> &t, std::vector<DirectionField> const &direction_fields,
    float unit, bool quiet = false)
{
    ceres::Problem problem;
    cv::Mat_<uint16_t> loss_status(state.size());

    int r_outer = radius+3;

    for(int oy=std::max(p[0]-r_outer,0);oy<=std::min(p[0]+r_outer,locs.rows-1);oy++)
        for(int ox=std::max(p[1]-r_outer,0);ox<=std::min(p[1]+r_outer,locs.cols-1);ox++)
            loss_status(oy,ox) = 0;

    for(int oy=std::max(p[0]-radius,0);oy<=std::min(p[0]+radius,locs.rows-1);oy++)
        for(int ox=std::max(p[1]-radius,0);ox<=std::min(p[1]+radius,locs.cols-1);ox++) {
            cv::Vec2i op = {oy, ox};
            if (cv::norm(p-op) <= radius)
                emptytrace_create_missing_centered_losses(problem, loss_status, op, state, locs, interp, t, direction_fields, unit);
        }
    for(int oy=std::max(p[0]-r_outer,0);oy<=std::min(p[0]+r_outer,locs.rows-1);oy++)
        for(int ox=std::max(p[1]-r_outer,0);ox<=std::min(p[1]+r_outer,locs.cols-1);ox++) {
            cv::Vec2i op = {oy, ox};
            if (cv::norm(p-op) > radius && problem.HasParameterBlock(&locs(op)[0]))
                problem.SetParameterBlockConstant(&locs(op)[0]);
        }


    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 10000;
    options.function_tolerance = 1e-4;
    options.use_nonmonotonic_steps = true;

//    if (problem.NumParameterBlocks() > 1) {
//        options.use_inner_iterations = true;
//    }
#ifdef VC_USE_CUDA_SPARSE
    // Check if Ceres was actually built with CUDA sparse support
    if (g_use_cuda) {
        if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::CUDA_SPARSE)) {
            options.linear_solver_type = ceres::SPARSE_SCHUR;
            options.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;

            if (options.linear_solver_type == ceres::SPARSE_SCHUR) {
                options.use_mixed_precision_solves = true;
            }
        } else {
            std::cerr << "Warning: use_cuda=true but Ceres was not built with CUDA sparse support. Falling back to CPU sparse." << std::endl;
        }
    }
#endif
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (!quiet)
        std::cout << "local solve radius " << radius << " " << summary.BriefReport() << std::endl;

    return sqrt(summary.final_cost/summary.num_residual_blocks);
}




template <typename E>
static E _max_d_ign(const E &a, const E &b)
{
    if (a == E(-1))
        return b;
    if (b == E(-1))
        return a;
    return std::max(a,b);
}

template <typename T, typename E>
static void _dist_iteration(T &from, T &to, int s)
{
    E magic = -1;
#pragma omp parallel for
    for(int k=0;k<s;k++)
        for(int j=0;j<s;j++)
            for(int i=0;i<s;i++) {
                E dist = from(k,j,i);
                if (dist == magic) {
                    if (k) dist = _max_d_ign(dist, from(k-1,j,i));
                    if (k < s-1) dist = _max_d_ign(dist, from(k+1,j,i));
                    if (j) dist = _max_d_ign(dist, from(k,j-1,i));
                    if (j < s-1) dist = _max_d_ign(dist, from(k,j+1,i));
                    if (i) dist = _max_d_ign(dist, from(k,j,i-1));
                    if (i < s-1) dist = _max_d_ign(dist, from(k,j,i+1));
                    if (dist != magic)
                        to(k,j,i) = dist+1;
                    else
                        to(k,j,i) = dist;
                }
                else
                    to(k,j,i) = dist;

            }
}

template <typename T, typename E>
static T distance_transform(const T &chunk, int steps, int size)
{
    T c1 = xt::empty<E>(chunk.shape());
    T c2 = xt::empty<E>(chunk.shape());

    c1 = chunk;

    E magic = -1;

    for(int n=0;n<steps/2;n++) {
        _dist_iteration<T,E>(c1,c2,size);
        _dist_iteration<T,E>(c2,c1,size);
    }

#pragma omp parallel for
    for(int z=0;z<size;z++)
        for(int y=0;y<size;y++)
            for(int x=0;x<size;x++)
                if (c1(z,y,x) == magic)
                    c1(z,y,x) = steps;

    return c1;
}

struct thresholdedDistance
{
    enum {BORDER = 16};
    enum {CHUNK_SIZE = 64};
    enum {FILL_V = 0};
    enum {TH = 170};
    const std::string UNIQUE_ID_STRING = "dqk247q6vz_"+std::to_string(BORDER)+"_"+std::to_string(CHUNK_SIZE)+"_"+std::to_string(FILL_V)+"_"+std::to_string(TH);
    template <typename T, typename E> void compute(const T &large, T &small, const cv::Vec3i &offset_large)
    {
        T outer = xt::empty<E>(large.shape());

        int s = CHUNK_SIZE+2*BORDER;
        E magic = -1;

        int good_count = 0;

#pragma omp parallel for
        for(int z=0;z<s;z++)
            for(int y=0;y<s;y++)
                for(int x=0;x<s;x++)
                    if (large(z,y,x) < TH)
                        outer(z,y,x) = magic;
                    else {
                        good_count++;
                        outer(z,y,x) = 0;
                    }

        outer = distance_transform<T,E>(outer, 15, s);

        int low = int(BORDER);
        int high = int(BORDER)+int(CHUNK_SIZE);

        auto crop_outer = view(outer, xt::range(low,high),xt::range(low,high),xt::range(low,high));

        small = crop_outer;
    }

};


QuadSurface *space_tracing_quad_phys(z5::Dataset *ds, float scale, ChunkCache *cache, cv::Vec3f origin, const nlohmann::json &params, const std::string &cache_root, float voxelsize, std::vector<DirectionField> const &direction_fields, QuadSurface* resume_surf)
{
    int stop_gen = params.value("generations", 100);
    float step = params.value("step_size", 20.0f);
    int rewind_gen = params.value("rewind_gen", -1);
    ALifeTime f_timer("empty space tracing\n");
    DSReader reader = {ds,scale,cache};

    int w, h;
    if (resume_surf) {
        cv::Mat resume_generations = resume_surf->channel("generations");
        double min_val, max_val;
        cv::minMaxLoc(resume_generations, &min_val, &max_val);
        int start_gen = (rewind_gen == -1) ? static_cast<int>(max_val) : rewind_gen;
        int gen_diff = stop_gen - start_gen;
        w = resume_generations.cols + 2 * gen_diff + 50;
        h = resume_generations.rows + 2 * gen_diff + 50;
    } else {
        // Calculate the maximum possible size the patch might grow to
        //FIXME show and handle area edge!
        w = 2*stop_gen+50;
        h = w;
    }
    cv::Size size = {w,h};
    cv::Rect bounds(0,0,w,h);

    int x0 = w/2;
    int y0 = h/2;

    // Together these represent the cached distance-transform of the thresholded surface volume
    thresholdedDistance compute;
    Chunked3d<uint8_t,thresholdedDistance> proc_tensor(compute, ds, cache, cache_root);

    // Debug: test the chunk cache by reading one voxel
    passTroughComputor pass;
    Chunked3d<uint8_t,passTroughComputor> dbg_tensor(pass, ds, cache);
    std::cout << "seed val " << origin << " " <<
    (int)dbg_tensor(origin[2],origin[1],origin[0]) << std::endl;

    auto timer = new ALifeTime("search & optimization ...");

    // This provides a cached interpolated version of the original surface volume
    CachedChunked3dInterpolator<uint8_t,thresholdedDistance> interp_global(proc_tensor);

    // fringe contains all 2D points around the edge of the patch where we might expand it
    // cands will contain new points adjacent to the fringe that are candidates to expand into
    std::vector<cv::Vec2i> fringe;
    std::vector<cv::Vec2i> cands;

    float T = step;
    float Ts = step*reader.scale;

    // The following track the state of the patch; they are each as big as the largest possible patch but initially empty
    // - locs defines the patch! It says for each 2D position, which 3D position it corresponds to
    // - state tracks whether each 2D position is part of the patch yet, and whether its 3D position has been found
    cv::Mat_<cv::Vec3d> locs(size,cv::Vec3f(-1,-1,-1));
    cv::Mat_<uint8_t> state(size,0);
    cv::Mat_<uint16_t> generations(size, (uint16_t)0);
    cv::Mat_<uint8_t> phys_fail(size,0);
    cv::Mat_<float> init_dist(size,0);
    cv::Mat_<uint16_t> loss_status(cv::Size(w,h),0);

    cv::Vec3f vx = {1,0,0};
    cv::Vec3f vy = {0,1,0};

    cv::Rect used_area;
    ceres::Problem big_problem;
    int loss_count = 0;
    int generation = 0;
    int succ = 0;  // number of quads successfully added to the patch (each of size approx. step**2)

    if (resume_surf) {
        float resume_step = 1.0 / resume_surf->scale()[0];
        if (std::abs(resume_step - step) > 1e-6) {
            throw std::runtime_error("Step size parameter mismatch between new trace and resume surface.");
        }

        cv::Mat_<cv::Vec3f> resume_points = resume_surf->rawPoints();
        cv::Mat resume_generations = resume_surf->channel("generations");

        int pad_x = (w - resume_points.cols) / 2;
        int pad_y = (h - resume_points.rows) / 2;

        used_area = cv::Rect(pad_x, pad_y, resume_points.cols, resume_points.rows);

        double min_val, max_val;
        cv::minMaxLoc(resume_generations, &min_val, &max_val);
        int start_gen = (rewind_gen == -1) ? static_cast<int>(max_val) : rewind_gen;
        generation = start_gen;

        for (int j = 0; j < resume_points.rows; ++j) {
            for (int i = 0; i < resume_points.cols; ++i) {
                int target_y = pad_y + j;
                int target_x = pad_x + i;
                uint16_t gen = resume_generations.at<uint16_t>(j, i);
                if (gen > 0 && gen <= start_gen) {
                    locs(target_y, target_x) = resume_points(j, i);
                    generations(target_y, target_x) = gen;
                    state(target_y, target_x) = STATE_LOC_VALID | STATE_COORD_VALID;
                    fringe.push_back({target_y, target_x});
                }
            }
        }

        for (const auto& p : fringe) {
            loss_count += emptytrace_create_missing_centered_losses(big_problem, loss_status, p, state, locs, interp_global, proc_tensor, direction_fields, Ts);
            succ++;
        }
        std::cout << "Resuming from generation " << generation << " with " << fringe.size() << " points. Initial loss count: " << loss_count << std::endl;

    } else {
        // Initialise the trace at the center of the available area, as a tiny single-quad patch at the seed point
        used_area = cv::Rect(x0,y0,2,2);
        //these are locations in the local volume!
        locs(y0,x0) = origin;
        locs(y0,x0+1) = origin+vx*0.1;
        locs(y0+1,x0) = origin+vy*0.1;
        locs(y0+1,x0+1) = origin+vx*0.1 + vy*0.1;

        state(y0,x0) = STATE_LOC_VALID | STATE_COORD_VALID;
        state(y0+1,x0) = STATE_LOC_VALID | STATE_COORD_VALID;
        state(y0,x0+1) = STATE_LOC_VALID | STATE_COORD_VALID;
        state(y0+1,x0+1) = STATE_LOC_VALID | STATE_COORD_VALID;

        // This Ceres problem is parameterised by locs; residuals are progressively added as the patch grows enforcing that
        // all points in the patch are correct distance in 2D vs 3D space, not too high curvature, near surface prediction, etc.

        // Add losses for every 'active' surface point (just the four currently) that doesn't yet have them
        loss_count += emptytrace_create_missing_centered_losses(big_problem, loss_status, {y0,x0}, state, locs, interp_global, proc_tensor, direction_fields, Ts);
        loss_count += emptytrace_create_missing_centered_losses(big_problem, loss_status, {y0+1,x0}, state, locs,  interp_global, proc_tensor, direction_fields, Ts);
        loss_count += emptytrace_create_missing_centered_losses(big_problem, loss_status, {y0,x0+1}, state, locs,  interp_global, proc_tensor, direction_fields, Ts);
        loss_count += emptytrace_create_missing_centered_losses(big_problem, loss_status, {y0+1,x0+1}, state, locs,  interp_global, proc_tensor, direction_fields, Ts);

        fringe.push_back({y0,x0});
        fringe.push_back({y0+1,x0});
        fringe.push_back({y0,x0+1});
        fringe.push_back({y0+1,x0+1});

        std::cout << "init loss count " << loss_count << std::endl;
    }

    ceres::Solver::Options options_big;
    options_big.linear_solver_type = ceres::SPARSE_SCHUR;
    options_big.use_nonmonotonic_steps = true;
#ifdef VC_USE_CUDA_SPARSE
    // Check if Ceres was actually built with CUDA sparse support
    if (g_use_cuda) {
        if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::CUDA_SPARSE)) {
            options_big.linear_solver_type = ceres::SPARSE_SCHUR;
            options_big.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;

            if (options_big.linear_solver_type == ceres::SPARSE_SCHUR) {
                options_big.use_mixed_precision_solves = true;
            }
        } else {
            std::cerr << "Warning: use_cuda=true but Ceres was not built with CUDA sparse support. Falling back to CPU sparse." << std::endl;
        }
    }
#endif
    options_big.minimizer_progress_to_stdout = false;
    options_big.max_num_iterations = 10000;

    // Solve the initial optimisation problem, just placing the first four vertices around the seed
    ceres::Solver::Summary big_summary;
    //just continue on resume no additional global opt
    if (!resume_surf) {
        ceres::Solve(options_big, &big_problem, &big_summary);
        std::cout << big_summary.BriefReport() << "\n";
    }

    // Prepare a new set of Ceres options used later during local solves
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 200;
    options.function_tolerance = 1e-3;


    std::vector<cv::Vec2i> neighs = {{1,0},{0,1},{-1,0},{0,-1}};

    int phys_fail_count = 0;

    int max_local_opt_r = 4;

    std::vector<float> gen_max_cost;
    std::vector<float> gen_avg_cost;

    int ref_max = 6;
    int curr_ref_min = ref_max;

    while (!fringe.empty()) {
        bool global_opt = generation <= 20;

        ALifeTime timer_gen;
        timer_gen.del_msg = "time per generation ";

        int phys_fail_count_gen = 0;
        generation++;
        if (stop_gen && generation >= stop_gen)
            break;

        std::vector<cv::Vec2i> rest_ps;  // contains candidates we didn't fully consider for some reason (write-only!)

        // For every point in the fringe (where we might expand the patch outwards), add to cands all
        // new 2D points we might add to the patch (and later find the corresponding 3D point for)
        for(const auto& p : fringe)
        {
            if ((state(p) & STATE_LOC_VALID) == 0) {
                if (state(p) & STATE_COORD_VALID)
                    for(const auto& n : neighs)
                        if (bounds.contains(cv::Point(p+n))
                            && (state(p+n) & (STATE_PROCESSING | STATE_LOC_VALID | STATE_COORD_VALID)) == 0) {
                            rest_ps.push_back(p+n);
                            }
                continue;
            }

            for(const auto& n : neighs)
                if (bounds.contains(cv::Point(p+n))
                    && (state(p+n) & STATE_PROCESSING) == 0
                    && (state(p+n) & STATE_LOC_VALID) == 0) {
                    state(p+n) |= STATE_PROCESSING;
                    cands.push_back(p+n);
                }
        }
        std::cout << "gen " << generation << " processing " << cands.size() << " fringe cands (total done " << succ << " fringe: " << fringe.size() << ")" << std::endl;
        fringe.resize(0);

        std::cout << "cands " << cands.size() << std::endl;

        if (generation % 10 == 0)
            curr_ref_min = std::min(curr_ref_min, 5);

        int succ_gen = 0;
        std::vector<cv::Vec2i> succ_gen_ps;

        // Build a structure that allows parallel iteration over cands, while avoiding any two threads simultaneously
        // considering two points that are too close to each other...
        OmpThreadPointCol cands_threadcol(max_local_opt_r*2+1, cands);

        // ...then start iterating over candidates in parallel using the above to yield points
#pragma omp parallel
        {
            CachedChunked3dInterpolator<uint8_t,thresholdedDistance> interp(proc_tensor);
//             int idx = rand() % cands.size();
            while (true) {
                int r2 = 2;
                int r = 1;
                double phys_fail_th = 0.1;
                cv::Vec2i p = cands_threadcol.next();
                if (p[0] == -1)
                    break;

                if (state(p) & (STATE_LOC_VALID | STATE_COORD_VALID))
                    continue;

                // p is now a 2D point we consider adding to the patch; find the best 3D point to map it to

                // Iterate all adjacent points that are in the patch, and find their 3D locations
                int ref_count = 0;
                cv::Vec3d avg = {0,0,0};
                std::vector<cv::Vec2i> srcs;
                for(int oy=std::max(p[0]-r,0);oy<=std::min(p[0]+r,locs.rows-1);oy++)
                    for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,locs.cols-1);ox++)
                        if (state(oy,ox) & STATE_LOC_VALID) {
                            ref_count++;
                            avg += locs(oy,ox);
                            srcs.push_back({oy,ox});
                        }

                // Of those adjacent points, find the one that itself has most adjacent in-patch points
                cv::Vec2i best_l = srcs[0];
                int best_ref_l = -1;
                int rec_ref_sum = 0;
                for(cv::Vec2i l : srcs) {
                    int ref_l = 0;
                    for(int oy=std::max(l[0]-r,0);oy<=std::min(l[0]+r,locs.rows-1);oy++)
                        for(int ox=std::max(l[1]-r,0);ox<=std::min(l[1]+r,locs.cols-1);ox++)
                            if (state(oy,ox) & STATE_LOC_VALID)
                                ref_l++;

                    rec_ref_sum += ref_l;

                    if (ref_l > best_ref_l) {
                        best_l = l;
                        best_ref_l = ref_l;
                    }
                }

                // Unused
                int ref_count2 = 0;
                for(int oy=std::max(p[0]-r2,0);oy<=std::min(p[0]+r2,locs.rows-1);oy++)
                    for(int ox=std::max(p[1]-r2,0);ox<=std::min(p[1]+r2,locs.cols-1);ox++)
                        // if (state(oy,ox) & (STATE_LOC_VALID | STATE_COORD_VALID)) {
                        if (state(oy,ox) & STATE_LOC_VALID) {
                            ref_count2++;
                        }

                // If the candidate 2D point is too 'loosely' connected to the patch, skip it; thus we prefer to keep the patch
                // compact rather than growing tendrils
                if (ref_count < 2 || ref_count+0.35*rec_ref_sum < curr_ref_min /*|| (generation > 3 && ref_count2 < 14)*/) {
                    state(p) &= ~STATE_PROCESSING;
#pragma omp critical
                    rest_ps.push_back(p);
                    continue;
                }

                // Initial guess for the corresponding 3D location is a perturbation of the position of the best-connected neighbor
                avg /= ref_count;
                cv::Vec3d init = locs(best_l)+cv::Vec3d((rand()%1000)/10000.0-0.05,(rand()%1000)/10000.0-0.05,(rand()%1000)/10000.0-0.05);
                locs(p) = init;

                // Set up a new local optimzation problem for the candidate point and its neighbors (initially just distance
                // and curvature losses, not nearness-to-surface)
                ceres::Problem problem;

                state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
                int local_loss_count = emptytrace_create_centered_losses(problem, p, state, locs, interp, proc_tensor, direction_fields, Ts);

                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);

                double loss1 = summary.final_cost;

                // If the solve couldn't find a good 3D position for the new point, try again with a different random initialisation (this time
                // around the average neighbor location instead of the best one)
                if (loss1 > phys_fail_th) {
                    cv::Vec3d best_loc = locs(p);
                    double best_loss = loss1;
                    for (int n=0;n<100;n++) {
                        int range = step*10;
                        locs(p) = avg + cv::Vec3d((rand()%(range*2))-range,(rand()%(range*2))-range,(rand()%(range*2))-range);
                        ceres::Solve(options, &problem, &summary);
                        loss1 = summary.final_cost;
                        if (loss1 < best_loss) {
                            best_loss = loss1;
                            best_loc = locs(p);
                        }
                        if (loss1 < phys_fail_th)
                            break;
                    }
                    loss1 = best_loss;
                    locs(p) = best_loc;
                }

                cv::Vec3d phys_only_loc = locs(p);
                // locs(p) = init;

                // Add to the local problem losses saying the new point should fall near the surface predictions
                gen_space_loss(problem, p, state, locs, proc_tensor);

                gen_space_line_loss(problem, p, {1,0}, state, locs, proc_tensor, T, 0.1, 100);
                gen_space_line_loss(problem, p, {-1,0}, state, locs, proc_tensor, T, 0.1, 100);
                gen_space_line_loss(problem, p, {0,1}, state, locs, proc_tensor, T, 0.1, 100);
                gen_space_line_loss(problem, p, {0,-1}, state, locs, proc_tensor, T, 0.1, 100);

                gen_direction_loss(problem, p, 1, state, locs, direction_fields);
                gen_direction_loss(problem, p, -1, state, locs, direction_fields);

                // Re-solve the updated local problem
                ceres::Solve(options, &problem, &summary);

                // Measure the worst-case distance from the surface predictions, of edges between the new point and its neighbors
                double dist;
                interp.Evaluate(locs(p)[2],locs(p)[1],locs(p)[0], &dist);
                int count = 0;
                for (auto &off : neighs) {
                    if (state(p+off) & STATE_LOC_VALID) {
                        for(int i=1;i<T;i++) {
                            float f1 = float(i)/T;
                            float f2 = 1-f1;
                            cv::Vec3d l = locs(p)*f1 + locs(p+off)*f2;
                            double d2;
                            interp.Evaluate(l[2],l[1],l[0], &d2);
                            dist = std::max(dist, d2);
                            count++;
                        }
                    }
                }

                init_dist(p) = dist;

                if (dist >= dist_th || summary.final_cost >= 0.1) {
                    // The solution to the local problem is bad -- large loss, or too far from the surface; still add to the global
                    // problem (as below) but don't mark the point location as valid
                    locs(p) = phys_only_loc;
                    state(p) = STATE_COORD_VALID;
                    if (global_opt) {
#pragma omp critical
                        loss_count += emptytrace_create_missing_centered_losses(big_problem, loss_status, p, state, locs,
                                                                                interp_global, proc_tensor, direction_fields, Ts, OPTIMIZE_ALL);
                    }
                    if (loss1 > phys_fail_th) {
                        // If even the first local solve (geometry only) was bad, try a less-local re-solve, that also adjusts the
                        // neighbors at progressively increasing radii as needed
                        phys_fail(p) = 1;

                        float err = 0;
                        for(int range = 1; range<=max_local_opt_r;range++) {
                            err = local_optimization(range, p, state, locs, interp, proc_tensor, direction_fields, Ts);
                            if (err <= phys_fail_th)
                                break;
                        }
                        if (err > phys_fail_th) {
                            std::cout << "local phys fail! " << err << std::endl;
#pragma omp atomic
                            phys_fail_count++;
#pragma omp atomic
                            phys_fail_count_gen++;
                        }
                    }
                }
                else {
                    // We found a good solution to the local problem; add losses for the new point to the global problem, add the
                    // new point to the fringe, and record as successful
                    generations(p) = generation;
                    if (global_opt) {
#pragma omp critical
                        loss_count += emptytrace_create_missing_centered_losses(big_problem, loss_status, p, state, locs,
                                                                                interp_global, proc_tensor, direction_fields, Ts);
                    }
#pragma omp atomic
                    succ++;
#pragma omp atomic
                    succ_gen++;
#pragma omp critical
                    {
                        if (!used_area.contains(cv::Point(p[1],p[0]))) {
                            used_area = used_area | cv::Rect(p[1],p[0],1,1);
                        }
                    }

#pragma omp critical
                    {
                        fringe.push_back(p);
                        succ_gen_ps.push_back(p);
                    }
                }
            }  // end parallel iteration over cands
        }

        // If there are now no fringe points, reduce the required compactness when considering cands for the next generation
        if (fringe.empty() && curr_ref_min > 2) {
            curr_ref_min--;
            std::cout << used_area << std::endl;
            for(int j=used_area.y;j<used_area.br().y;j++)
                for(int i=used_area.x;i<used_area.br().x;i++) {
                    if (state(j, i) & STATE_LOC_VALID)
                        fringe.push_back({j,i});
                }
            std::cout << "new limit " << curr_ref_min << " " << fringe.size() << std::endl;
        }
        else if (!fringe.empty())
            curr_ref_min = ref_max;

        for(const auto& p: fringe)
            if (locs(p)[0] == -1)
                std::cout << "impossible! " << p << " " << cv::Vec2i(y0,x0) << std::endl;

        if (generation >= 3) {
            options_big.max_num_iterations = 10;
        }

        //this actually did work (but was slow ...)
        if (phys_fail_count_gen) {
            options_big.minimizer_progress_to_stdout = true;
            options_big.max_num_iterations = 100;
        }
        else
            options_big.minimizer_progress_to_stdout = false;

        if (!global_opt) {
            // For late generations, instead of re-solving the global problem, solve many local-ish problems, around each
            // of the newly added points
            std::vector<cv::Vec2i> opt_local;
            for(auto p : succ_gen_ps)
                if (p[0] % 4 == 0 && p[1] % 4 == 0)
                    opt_local.push_back(p);

            if (!opt_local.empty()) {
                OmpThreadPointCol opt_local_threadcol(17, opt_local);

#pragma omp parallel
                while (true)
                {
                    CachedChunked3dInterpolator<uint8_t,thresholdedDistance> interp(proc_tensor);
                    cv::Vec2i p = opt_local_threadcol.next();
                    if (p[0] == -1)
                        break;

                    local_optimization(8, p, state, locs, interp, proc_tensor, direction_fields, Ts, true);
                }
            }
        }
        else {
            // For early generations, re-solve the big problem, jointly optimising the locations of all points in the patch
            std::cout << "running big solve" << std::endl;
            ceres::Solve(options_big, &big_problem, &big_summary);
            std::cout << big_summary.BriefReport() << "\n";
            std::cout << "avg err: " << sqrt(big_summary.final_cost/big_summary.num_residual_blocks) << std::endl;
        }

        if (generation > 10 && global_opt) {
            // Beyond 10 generations but while still trying global re-solves, simplify the big problem by fixing locations
            // of points that are already 'certain', in the sense they are not near any other points that don't yet have valid
            // locations
            cv::Mat_<cv::Vec2d> _empty;
            freeze_inner_params(big_problem, 10, state, locs, _empty, loss_status, STATE_LOC_VALID | STATE_COORD_VALID);
        }

        cands.resize(0);

        // Record the cost of the current patch, by re-evaluating all losses within the patch bbox region
        cv::Rect used_area_safe = used_area;
        used_area_safe.x -= 2;
        used_area_safe.y -= 2;
        used_area_safe.width += 4;
        used_area_safe.height += 4;
        cv::Mat_<cv::Vec3d> locs_crop = locs(used_area_safe);
        cv::Mat_<uint8_t> state_crop = state(used_area_safe);
        double max_cost = 0;
        double avg_cost = 0;
        int cost_count = 0;
        for(int j=2;j<locs_crop.rows-2;j++)
            for(int i=2;i<locs_crop.cols-2;i++) {
                ceres::Problem problem;
                emptytrace_create_centered_losses(problem, {j,i}, state_crop, locs_crop, interp_global, proc_tensor, direction_fields, Ts);
                double cost = 0.0;
                problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
                max_cost = std::max(max_cost, cost);
                avg_cost += cost;
                cost_count++;
            }
        gen_avg_cost.push_back(avg_cost/cost_count);
        gen_max_cost.push_back(max_cost);

        float const current_area_vx2 = double(succ)*step*step;
        float const current_area_cm2 = current_area_vx2 * voxelsize * voxelsize / 1e8;
        printf("-> total done %d/ fringe: %ld surf: %fG vx^2 (%f cm^2)\n", succ, (long)fringe.size(), current_area_vx2/1e9, current_area_cm2);

        timer_gen.unit = succ_gen*step*step;
        timer_gen.unit_string = "vx^2";
        print_accessor_stats();

    }  // end while fringe is non-empty
    delete timer;

    cv::Rect used_area_safe = used_area;
    used_area_safe.x -= 2;
    used_area_safe.y -= 2;
    used_area_safe.width += 4;
    used_area_safe.height += 4;
    locs = locs(used_area_safe);
    state = state(used_area_safe);
    generations = generations(used_area_safe);

    double max_cost = 0;
    double avg_cost = 0;
    int count = 0;
    for(int j=2;j<locs.rows-2;j++)
        for(int i=2;i<locs.cols-2;i++) {
            ceres::Problem problem;
            emptytrace_create_centered_losses(problem, {j,i}, state, locs, interp_global, proc_tensor, direction_fields, Ts);
            double cost = 0.0;
            problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
            max_cost = std::max(max_cost, cost);
            avg_cost += cost;
            count++;
        }
    avg_cost /= count;

    float const area_est_vx2 = succ*step*step;
    float const area_est_cm2 = area_est_vx2 * voxelsize * voxelsize / 1e8;
    printf("generated approximate surface %f vx^2 (%f cm^2)\n", area_est_vx2, area_est_cm2);

    auto surf = new QuadSurface(locs, {1/T, 1/T});
    surf->setChannel("generations", generations);

    surf->meta = new nlohmann::json;
    (*surf->meta)["area_vx2"] = area_est_vx2;
    (*surf->meta)["area_cm2"] = area_est_cm2;
    (*surf->meta)["max_cost"] = max_cost;
    (*surf->meta)["avg_cost"] = avg_cost;
    (*surf->meta)["max_gen"] = generation;
    (*surf->meta)["gen_avg_cost"] = gen_avg_cost;
    (*surf->meta)["gen_max_cost"] = gen_max_cost;
    (*surf->meta)["seed"] = {origin[0],origin[1],origin[2]};
    (*surf->meta)["elapsed_time_s"] = f_timer.seconds();

    return surf;
}
