#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/SurfaceModeling.hpp"
#include "vc/core/types/ChunkedTensor.hpp"

#include "z5/factory.hxx"
#include <nlohmann/json.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h>



using json = nlohmann::json;

static int trace_mul;
static float dist_w;
static float straight_w;
static float surf_w;
static float z_loc_loss_w;
static float wind_w;
static float wind_th;
static int inpaint_back_range;

static float normal_w = 0.3;

static inline cv::Vec2f mul(const cv::Vec2f &a, const cv::Vec2f &b)
{
    return{a[0]*b[0],a[1]*b[1]};
}

static float dot_s(const cv::Vec3f &p)
{
    return p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
}

template <typename E>
float ldist(const E &p, const cv::Vec3f &tgt_o, const cv::Vec3f &tgt_v)
{
    return cv::norm((p-tgt_o).cross(p-tgt_o-tgt_v))/cv::norm(tgt_v);
}

template <typename E>
static float search_min_line(const cv::Mat_<E> &points, cv::Vec2f &loc, cv::Vec3f &out, cv::Vec3f tgt_o, cv::Vec3f tgt_v, cv::Vec2f init_step, float min_step_x)
{
    cv::Rect boundary(1,1,points.cols-2,points.rows-2);
    if (!boundary.contains(cv::Point(loc))) {
        out = {-1,-1,-1};
        return -1;
    }
    
    bool changed = true;
    E val = at_int(points, loc);
    out = val;
    float best = ldist(val, tgt_o, tgt_v);
    float res;
    
    //TODO check maybe add more search patterns, compare motion estimatino for video compression, x264/x265, ...
    std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,-1},{-1,0},{-1,1},{1,-1},{1,0},{1,1}};
    // std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,0},{1,0}};
    cv::Vec2f step = init_step;
    
    while (changed) {
        changed = false;
        
        for(auto &off : search) {
            cv::Vec2f cand = loc+mul(off,step);
            
            //just skip if out of bounds
            if (!boundary.contains(cv::Point(cand)))
                continue;
                
                val = at_int(points, cand);
                res = ldist(val, tgt_o, tgt_v);
                if (res < best) {
                    changed = true;
                    best = res;
                    loc = cand;
                    out = val;
                }
        }
        
        if (changed)
            continue;
        
        step *= 0.5;
        changed = true;
        
        if (step[0] < min_step_x)
            break;
    }
    
    return best;
}

template <typename E>
float line_off(const E &p, const cv::Vec3f &tgt_o, const cv::Vec3f &tgt_v)
{
    return (tgt_o-p).dot(tgt_v)/dot_s(tgt_v);
}

//l is [y, x]!
bool loc_valid_nan(const cv::Mat_<float> &m, const cv::Vec2d &l)
{
    if (std::isnan(l[0]))
        return false;
    
    cv::Rect bounds = {0, 0, m.rows-2,m.cols-2};
    cv::Vec2i li = {static_cast<int>(floor(l[0])),static_cast<int>(floor(l[1]))};
    
    if (!bounds.contains(cv::Point(li)))
        return false;
    
    if (std::isnan(m(li[0],li[1])))
        return false;
    if (std::isnan(m(li[0]+1,li[1])))
        return false;
    if (std::isnan(m(li[0],li[1]+1)))
        return false;
    if (std::isnan(m(li[0]+1,li[1]+1)))
        return false;
    return true;
}

bool loc_valid_nan_xy(const cv::Mat_<float> &m, const cv::Vec2d &l)
{
    return loc_valid_nan(m, {l[1],l[0]});
}

float find_wind_x(cv::Mat_<float> &winding, cv::Vec2f &loc, float tgt_wind)
{
    if (!loc_valid_nan_xy(winding, loc))
        return -1;
    
    float best_diff = abs(at_int(winding,loc)-tgt_wind);
    
    std::vector<cv::Vec2f> neighs = {{1,0},{-1,0}};
    
    float step = 16.0;
    bool updated = true;
    while (updated)
    {
        updated = false;
        for (auto n : neighs) {
            cv::Vec2f cand = loc + step*n;
            if (!loc_valid_nan_xy(winding, cand))
                continue;
            float diff = abs(at_int(winding,cand)-tgt_wind);
            if (diff < best_diff) {
                best_diff = diff;
                loc = cand;
                updated = true;
                break;
            }
        }
        
        if (!updated) {
            if (step <= 0.001)
                break;
            step /= 2;
            updated = true;
        }
    }

    return best_diff;
}

std::string time_str()
{
    using namespace std::chrono;
    auto now = system_clock::now();
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    auto timer = system_clock::to_time_t(now);
    std::tm bt = *std::localtime(&timer);
    
    std::ostringstream oss;
    oss << std::put_time(&bt, "%Y%m%d%H%M%S");
    oss << std::setfill('0') << std::setw(3) << ms.count();
    
    return oss.str();
}

int gen_surfloss(const cv::Vec2i p, ceres::Problem &problem, const cv::Mat_<uint8_t> &state, const cv::Mat_<cv::Vec3f> &points_in, 
    cv::Mat_<cv::Vec3d> &points, cv::Mat_<cv::Vec2d> &locs, float w = 0.1, int flags = 0)
{
    if ((state(p) & STATE_LOC_VALID) == 0)
        return 0;

    problem.AddResidualBlock(SurfaceLossD::Create(points_in, w), nullptr, &points(p)[0], &locs(p)[0]);

    return 1;
}

//gen straigt loss given point and 3 offsets
int gen_dist_loss_fill(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &dpoints, 
    float unit, bool optimize_all, ceres::ResidualBlockId *res, float w)
{
    if ((state(p) & (STATE_LOC_VALID|STATE_COORD_VALID)) == 0)
        return 0;
    if ((state(p+off) & (STATE_LOC_VALID|STATE_COORD_VALID)) == 0)
        return 0;
    
    ceres::ResidualBlockId tmp = problem.AddResidualBlock(DistLoss::Create(unit*cv::norm(off),w), nullptr, &dpoints(p)[0], &dpoints(p+off)[0]);
    
    if (res)
        *res = tmp;
    
    if (!optimize_all)
        problem.SetParameterBlockConstant(&dpoints(p+off)[0]);
    
    return 1;
}

static bool loc_valid(int state)
{
    return state & STATE_LOC_VALID;
}

static bool coord_valid(int state)
{
    return (state & STATE_COORD_VALID) || (state & STATE_LOC_VALID);
}

template <typename E>
struct SampledZNormalLoss {
    SampledZNormalLoss(const cv::Mat_<E> &normals, float normal_loc_x, float mul_z, float w) : _normals(normals), _normal_loc_x(normal_loc_x),_mul_z(mul_z), _w(w) {};
    template <typename T>
    //p1 is the central point, p2 a neighboring point to calculate a direction against p1
    bool operator()(const T* const p1, const T* const p2, T* residual) const {        
        if (!loc_valid(_normals, {val(p1[2])*_mul_z, _normal_loc_x})) {
            residual[0] = T(0);
            return true;
        }
        
        T n[3];
        interp_lin_2d(_normals, p1[2]*T(_mul_z), T(_normal_loc_x), n);
        
        T v[3];
        v[0] = p2[0]-p1[0];
        v[1] = p2[1]-p1[1];
        v[2] = p2[2]-p1[2];
        
        T dot;
        dot = v[0]*n[0] + v[1]*n[1] + v[2]*n[2];
        
        T la = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        T lb = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
        
        if (la <= 0.0 || lb <= 0.0) {
            residual[0] = T(0);
            return true;
        }
        
        residual[0] = T(_w)*dot/(la*lb);
        
        return true;
    }
    
    const cv::Mat_<E> &_normals;
    float _normal_loc_x, _mul_z, _w;
    
    static ceres::CostFunction* Create(const cv::Mat_<E> &normals, float normal_loc_x, float mul_z, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<SampledZNormalLoss, 1, 3, 3>(new SampledZNormalLoss(normals, normal_loc_x, mul_z, w));
    }
    
};

//gen straigt loss given point and 3 offsets
int gen_normal_loss_fill(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points, 
    cv::Mat_<cv::Vec3f> &normals, float tgt_wind_x, float mul_z, bool optimize_all, ceres::ResidualBlockId *res, float w)
{
    if ((state(p) & (STATE_LOC_VALID|STATE_COORD_VALID)) == 0)
        return 0;
    if ((state(p+off) & (STATE_LOC_VALID|STATE_COORD_VALID)) == 0)
        return 0;
    
    ceres::ResidualBlockId tmp = problem.AddResidualBlock(SampledZNormalLoss<cv::Vec3f>::Create(normals,tgt_wind_x,mul_z,w), 
                                                          new ceres::CauchyLoss(1.0), &points(p)[0], &points(p+off)[0]);
    
    if (res)
        *res = tmp;
    
    if (!optimize_all)
        problem.SetParameterBlockConstant(&points(p+off)[0]);
    
    return 1;
}

//gen straigt loss given point and 3 offsets
int gen_straight_loss2(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &o1, const cv::Vec2i &o2, const cv::Vec2i &o3, 
    cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &dpoints, bool optimize_all, float w)
{
    if (!coord_valid(state(p+o1)))
        return 0;
    if (!coord_valid(state(p+o2)))
        return 0;
    if (!coord_valid(state(p+o3)))
        return 0;
    
    problem.AddResidualBlock(StraightLoss2::Create(w), nullptr, &dpoints(p+o1)[0], &dpoints(p+o2)[0], &dpoints(p+o3)[0]);
    // problem.AddResidualBlock(StraightLoss2::Create(w), new ceres::CauchyLoss(1.0), &dpoints(p+o1)[0], &dpoints(p+o2)[0], &dpoints(p+o3)[0]);
    
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

int create_centered_losses_left_large(ceres::Problem &problem, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, 
    const cv::Mat_<cv::Vec3f> &points_in, cv::Mat_<cv::Vec3d> &points, cv::Mat_<cv::Vec2d> &locs, cv::Mat_<cv::Vec3f> &normals, 
    float tgt_wind_x, float mul_z, float unit, int flags = 0)
{
    if (!coord_valid(state(p)))
        return 0;
    
    //generate losses for point p
    int count = 0;
    
    //horizontal
    count += gen_straight_loss2(problem, p, {0,-2},{0,-1},{0,0}, state, points, flags & OPTIMIZE_ALL, straight_w);
    
    //further and diag!
    count += gen_straight_loss2(problem, p, {-2,-2},{-1,-1},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.7*straight_w);
    count += gen_straight_loss2(problem, p, {2,-2},{1,-1},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.7*straight_w);
    
    count += gen_straight_loss2(problem, p, {-4,-2},{-2,-1},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.5*straight_w);
    count += gen_straight_loss2(problem, p, {4,-2},{2,-1},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.5*straight_w);
    
    count += gen_straight_loss2(problem, p, {-6,-2},{-2,-1},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.5*straight_w);
    count += gen_straight_loss2(problem, p, {6,-2},{3,-1},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.5*straight_w);
  
    count += gen_straight_loss2(problem, p, {0,-4},{0,-2},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.7*straight_w);
    count += gen_straight_loss2(problem, p, {2,-4},{1,-2},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.7*straight_w);
    count += gen_straight_loss2(problem, p, {-2,-4},{-1,-2},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.7*straight_w);
    count += gen_straight_loss2(problem, p, {4,-4},{2,-2},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.7*straight_w);
    count += gen_straight_loss2(problem, p, {-4,-4},{-2,-2},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.7*straight_w);
    
    //direct neighboars
    count += gen_dist_loss_fill(problem, p, {0,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    
    //diagonal neighbors
    count += gen_dist_loss_fill(problem, p, {1,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, 0.7*dist_w);
    count += gen_dist_loss_fill(problem, p, {-1,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, 0.7*dist_w);
    
    
    count += gen_dist_loss_fill(problem, p, {2,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, 0.5*dist_w);
    count += gen_dist_loss_fill(problem, p, {-2,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, 0.5*dist_w);
    count += gen_dist_loss_fill(problem, p, {4,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, 0.5*dist_w);
    count += gen_dist_loss_fill(problem, p, {-4,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, 0.5*dist_w);
    
    //far left
    count += gen_dist_loss_fill(problem, p, {0,-2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, 0.5*dist_w);
    count += gen_dist_loss_fill(problem, p, {1,-2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, 0.5*dist_w);
    count += gen_dist_loss_fill(problem, p, {-1,-2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, 0.5*dist_w);
    count += gen_dist_loss_fill(problem, p, {2,-2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, 0.5*dist_w);
    count += gen_dist_loss_fill(problem, p, {-2,-2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, 0.5*dist_w);
    
    if (flags & LOSS_ON_SURF)
        gen_surfloss(p, problem, state, points_in, points, locs, surf_w);
    
    if (flags & LOSS_ON_NORMALS) {
        count += gen_normal_loss_fill(problem, p, {0,-1}, state, points, normals, tgt_wind_x, mul_z, flags & OPTIMIZE_ALL, nullptr, normal_w);
        count += gen_normal_loss_fill(problem, p, {1,-1}, state, points, normals, tgt_wind_x, mul_z, flags & OPTIMIZE_ALL, nullptr, normal_w);
        count += gen_normal_loss_fill(problem, p, {-1,-1}, state, points, normals, tgt_wind_x, mul_z, flags & OPTIMIZE_ALL, nullptr, normal_w);
    }

    return count;
}

int create_centered_losses(ceres::Problem &problem, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, 
    const cv::Mat_<cv::Vec3f> &points_in, cv::Mat_<cv::Vec3d> &points, cv::Mat_<cv::Vec2d> &locs, cv::Mat_<cv::Vec3f> &normals, 
    float tgt_wind_x, float mul_z, float unit, int flags = 0, float w_mul = 1.0)
{
    if (!coord_valid(state(p)))
        return 0;

    //generate losses for point p
    int count = 0;
    
    //horizontal
    count += gen_straight_loss2(problem, p, {0,-1},{0,0},{0,1}, state, points, flags & OPTIMIZE_ALL, straight_w*w_mul);
    
    //vertical
    count += gen_straight_loss2(problem, p, {-1,0},{0,0},{1,0}, state, points, flags & OPTIMIZE_ALL, straight_w*w_mul);
    
    //diag
    count += gen_straight_loss2(problem, p, {-1,-1},{0,0},{1,1}, state, points, flags & OPTIMIZE_ALL, straight_w*w_mul);
    count += gen_straight_loss2(problem, p, {-1,1},{0,0},{1,-1}, state, points, flags & OPTIMIZE_ALL, straight_w*w_mul);
    count += gen_straight_loss2(problem, p, {1,-1},{0,0},{-1,1}, state, points, flags & OPTIMIZE_ALL, straight_w*w_mul);
    count += gen_straight_loss2(problem, p, {1,1},{0,0},{-1,-1}, state, points, flags & OPTIMIZE_ALL, straight_w*w_mul);
    
    
    //direct neighboars
    count += gen_dist_loss_fill(problem, p, {0,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    count += gen_dist_loss_fill(problem, p, {0,1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    count += gen_dist_loss_fill(problem, p, {-1,0}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    count += gen_dist_loss_fill(problem, p, {1,-0}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    
    //diag neighboars
    count += gen_dist_loss_fill(problem, p, {1,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    count += gen_dist_loss_fill(problem, p, {1,1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    count += gen_dist_loss_fill(problem, p, {-1,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    count += gen_dist_loss_fill(problem, p, {-1,1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    
    //+2
    count += gen_dist_loss_fill(problem, p, {-2,0}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    count += gen_dist_loss_fill(problem, p, {2,0}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    count += gen_dist_loss_fill(problem, p, {0,-2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    count += gen_dist_loss_fill(problem, p, {0,2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    
    //+2 diag
    count += gen_dist_loss_fill(problem, p, {2,-2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    count += gen_dist_loss_fill(problem, p, {2,2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    count += gen_dist_loss_fill(problem, p, {-2,-2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    count += gen_dist_loss_fill(problem, p, {-2,2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    
    
    if (flags & LOSS_ON_SURF)
        gen_surfloss(p, problem, state, points_in, points, locs, surf_w*w_mul);
    
    
    if (flags & LOSS_ON_NORMALS) {
        count += gen_normal_loss_fill(problem, p, {0,-1}, state, points, normals, tgt_wind_x, mul_z, flags & OPTIMIZE_ALL, nullptr, normal_w*w_mul);
        count += gen_normal_loss_fill(problem, p, {0,1}, state, points, normals, tgt_wind_x, mul_z, flags & OPTIMIZE_ALL, nullptr, normal_w*w_mul);
        count += gen_normal_loss_fill(problem, p, {1,0}, state, points, normals, tgt_wind_x, mul_z, flags & OPTIMIZE_ALL, nullptr, normal_w*w_mul);
        count += gen_normal_loss_fill(problem, p, {-1,0}, state, points, normals, tgt_wind_x, mul_z, flags & OPTIMIZE_ALL, nullptr, normal_w*w_mul);
    }
    
    return count;
}

float find_loc_wind_slow(cv::Vec2f &loc, float tgt_wind, const cv::Mat_<cv::Vec3f> &points, const cv::Mat_<float> &winding, const cv::Vec3f &tgt, float th)
{
    float best_res = -1;
    uint32_t sr = loc[0]+loc[1];
    for(int r=0;r<1000;r++) {
        cv::Vec2f cand = loc;
        
        if (r)
            cand = {static_cast<float>(rand_r(&sr) % points.cols), static_cast<float>(rand_r(&sr) % points.rows)};
        
        if (std::isnan(winding(cand[1],cand[0])) || abs(winding(cand[1],cand[0])-tgt_wind) > 0.5)
            continue;
        
        cv::Vec3f out_;
        float res = min_loc(points, cand, out_, {tgt}, {0}, nullptr, 4.0, 0.01);
        
        if (res < 0)
            continue;
        
        if (std::isnan(winding(cand[1],cand[0])) || abs(winding(cand[1],cand[0])-tgt_wind) > 0.3)
            continue;
        
        if (res < th) {
            loc = cand;
            return res;
        }
        
        if (best_res == -1 || res < best_res) {
            loc = cand;
            best_res = res;
        }
    }
    
    return sqrt(best_res);
}

static float sdist(const cv::Vec3f &a, const cv::Vec3f &b)
{
    cv::Vec3f d = a-b;
    return d.dot(d);
}

float min_loc_wind(const cv::Mat_<cv::Vec3f> &points, const cv::Mat_<float> &winding, cv::Vec2f &loc, cv::Vec3f &out, 
    float tgt_wind, const cv::Vec3f &tgt, float init_step, float min_step, bool avoid_edges = true)
{
    if (!loc_valid(points, {loc[1],loc[0]})) {
        out = {-1,-1,-1};
        return -1;
    }
    
    bool changed = true;
    cv::Vec3f val = at_int(points, loc);
    out = val;
    float best = abs(at_int(winding, loc)-tgt_wind)*10 + sdist(val,tgt);
    float res;
    
    std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,0},{1,0}};
    float step = init_step;
    
    while (changed) {
        changed = false;
        
        for(auto &off : search) {
            cv::Vec2f cand = loc+off*step;
            
            if (!loc_valid(points, {cand[1],cand[0]})) {
                if (avoid_edges)
                    continue;
                else if (step < min_step*2) 
                {
                    out = {-1,-1,-1};
                    return -1;
                }
            }
            
            val = at_int(points, cand);
            // std::cout << "at" << cand << val << std::endl;
            res = abs(at_int(winding, cand)-tgt_wind)*10;
            res += sdist(val,tgt);
            if (res < best) {
                changed = true;
                best = res;
                loc = cand;
                out = val;
            }
        }
        
        if (changed)
            continue;
        
        step *= 0.5;
        changed = true;
        
        if (step < min_step)
            break;
    }
    
    return best;
}

float find_loc_wind(cv::Vec2f &loc, float tgt_wind, const cv::Mat_<cv::Vec3f> &points, const cv::Mat_<float> &winding, 
    const cv::Vec3f &tgt, float th, bool avoid_edges = true)
{
    float best_res = -1;
    uint32_t sr = loc[0]+loc[1];
    for(int r=0,full_r=0;r<1000,full_r<10000;full_r++) {
        cv::Vec2f cand = loc;
        
        if (full_r || !loc_valid_nan_xy(winding, cand))
            cand = {static_cast<float>(rand_r(&sr) % points.cols), static_cast<float>(rand_r(&sr) % points.rows)};
        
        if (abs(winding(cand[1],cand[0])-tgt_wind) > 0.3)
            continue;
        
        r++;
        
        cv::Vec3f out_;
        float res = min_loc_wind(points, winding, cand, out_, tgt_wind, tgt, 4.0, 0.001, avoid_edges);
        
        if (res < 0)
            continue;
        
        if (abs(winding(cand[1],cand[0])-tgt_wind) > 0.3)
            continue;
        
        if (res < th) {
            loc = cand;
            return res;
        }
        
        if (best_res == -1 || res < best_res) {
            loc = cand;
            best_res = res;
        }
    }
    
    return best_res;
}

template<typename T, typename E>
void interp_lin_2d(const cv::Mat_<E> &m, T y, T x, T *v) {
    if (val(y) < 0)
        y = T(0);
    if (val(y) > m.rows-2)
        y = T(m.rows-2);
    
    if (val(x) < 0)
        x = T(0);
    if (val(x) > m.cols-2)
        x = T(m.cols-2);
    
    int yi = val(y);
    int xi = val(x);
    
    T fx = x - T(xi);
    T fy = y - T(yi);
    
    E c00 = m(yi,xi);
    E c01 = m(yi,xi+1);
    E c10 = m(yi+1,xi);
    E c11 = m(yi+1,xi+1);
    
    T c0 = (T(1)-fx)*T(c00) + fx*T(c01);
    T c1 = (T(1)-fx)*T(c10) + fx*T(c11);
    v[0] = (T(1)-fy)*c0 + fy*c1;
}

static bool loc_valid(const cv::Mat_<float> &m, const cv::Vec2d &l)
{
    if (l[0] == -1)
        return false;
    
    cv::Rect bounds = {0, 0, m.rows-2,m.cols-2};
    cv::Vec2i li = {static_cast<int>(floor(l[0])),static_cast<int>(floor(l[1]))};
    
    if (!bounds.contains(cv::Point(li)))
        return false;
    
    if (std::isnan(m(li[0],li[1])))
        return false;
    if (std::isnan(m(li[0]+1,li[1])))
        return false;
    if (std::isnan(m(li[0],li[1]+1)))
        return false;
    if (std::isnan(m(li[0]+1,li[1]+1)))
        return false;
    return true;
}

template<typename E>
//cost functions for physical paper
struct Interp2DLoss {
    //NOTE we expect loc to be [y, x]
    Interp2DLoss(const cv::Mat_<E> &m, E tgt, float w) : _m(m), _w(w), _tgt(tgt) {};
    template <typename T>
    bool operator()(const T* const l, T* residual) const {
        T v[3];
        
        if (!loc_valid(_m, {val(l[0]), val(l[1])})) {
            residual[0] = T(0);
            return true;
        }
        
        interp_lin_2d(_m, l[0], l[1], v);
        
        residual[0] = T(_w)*(v[0] - T(_tgt));
        
        return true;
    }
    
    const cv::Mat_<E> &_m;
    float _w;
    E _tgt;
    
    static ceres::CostFunction* Create(const cv::Mat_<E> &m, E tgt, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<Interp2DLoss, 1, 2>(new Interp2DLoss(m, tgt, w));
    }
    
};

cv::Mat_<cv::Vec3f> points_hr_grounding(const cv::Mat_<uint8_t> &state, std::vector<float> tgt_wind, const cv::Mat_<float> &winding, 
    const cv::Mat_<cv::Vec3f> &points_tgt_in, const cv::Mat_<cv::Vec3f> &points_src, int step)
{
    cv::Mat_<cv::Vec3f> points_hr(points_tgt_in.rows*step, points_tgt_in.cols*step, {0,0,0});
    cv::Mat_<int> counts_hr(points_tgt_in.rows*step, points_tgt_in.cols*step, 0);
    
    cv::Mat_<cv::Vec3f> points_tgt = points_tgt_in.clone();

    for(int j=0;j<points_tgt.rows-1;j++)
        for(int i=0;i<points_tgt.cols-1;i++) {
            if (points_tgt(j,i)[0] == -1)
                continue;
            if (points_tgt(j,i+1)[0] == -1)
                continue;
            if (points_tgt(j+1,i)[0] == -1)
                continue;
            if (points_tgt(j+1,i+1)[0] == -1)
                continue;
            
            cv::Vec2f l00, l01, l10, l11;
            
            float hr_th = 20.0;
            float res;
            cv::Vec3f out_;

            res = find_loc_wind_slow(l00, tgt_wind[i], points_src, winding, points_tgt(j,i), hr_th*hr_th);
            if (res < 0 || res > hr_th*hr_th)
                continue;

            l01 = l00;
            res = min_loc(points_src, l01, out_, {points_tgt(j,i+1)}, {0}, nullptr, 1.0, 0.01);
            if (res < 0 || res > hr_th*hr_th)
                continue;
            l10 = l00;
            res = min_loc(points_src, l10, out_, {points_tgt(j+1,i)}, {0}, nullptr, 1.0, 0.01);
            if (res < 0 || res > hr_th*hr_th)
                continue;
            l11 = l00;
            res = min_loc(points_src, l11, out_, {points_tgt(j+1,i+1)}, {0}, nullptr, 1.0, 0.01);
            if (res < 0 || res > hr_th*hr_th)
                continue;
            
            //FIXME should also re-use already found corners for interpolation!
            points_tgt(j, i) = at_int(points_src, l00);
            points_tgt(j, i+1) = at_int(points_src, l01);
            points_tgt(j+1, i) = at_int(points_src, l10);
            points_tgt(j+1, i+1) = at_int(points_src, l11);
            
            l00 = {l00[1],l00[0]};
            l01 = {l01[1],l01[0]};
            l10 = {l10[1],l10[0]};
            l11 = {l11[1],l11[0]};            
            
            // std::cout << "succ!" << res << cv::Vec2i(i,j) << l00 << l01 << points_tgt(j,i) << std::endl;
            
            for(int sy=0;sy<=step;sy++)
                for(int sx=0;sx<=step;sx++) {
                    float fx = float(sx)/step;
                    float fy = float(sy)/step;
                    cv::Vec2f l0 = (1-fx)*l00 + fx*l01;
                    cv::Vec2f l1 = (1-fx)*l10 + fx*l11;
                    cv::Vec2f l = (1-fy)*l0 + fy*l1;
                    // std::cout << l << at_int(points_src, {l[1],l[0]}) << std::endl;
                    points_hr(j*step+sy,i*step+sx) += at_int(points_src, {l[1],l[0]});
                    counts_hr(j*step+sy,i*step+sx) += 1;
                }
        }

        for(int j=0;j<points_tgt.rows-1;j++)
            for(int i=0;i<points_tgt.cols-1;i++) {
                if (!counts_hr(j*step+1,i*step+1)) {
                    for(int sy=0;sy<=step;sy++)
                        for(int sx=0;sx<=step;sx++) {
                            float fx = float(sx)/step;
                            float fy = float(sy)/step;
                            cv::Vec3f c0 = (1-fx)*points_tgt(j,i) + fx*points_tgt(j,i+1);
                            cv::Vec3f c1 = (1-fx)*points_tgt(j+1,i) + fx*points_tgt(j+1,i+1);
                            cv::Vec3f c = (1-fy)*c0 + fy*c1;
                            // std::cout << l << at_int(points_src, {l[1],l[0]}) << std::endl;
                            points_hr(j*step+sy,i*step+sx) += c;
                            counts_hr(j*step+sy,i*step+sx) += 1;
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

template <typename E> cv::Mat_<E> pad(const cv::Mat_<E> &src, int amount, const E &val)
{
    cv::Mat_<E> m(src.rows+2*amount, src.cols+2*amount, val);
    
    src.copyTo(m(cv::Rect(0, amount, src.cols, src.rows)));
    
    std::cout << "pad " << src.size() << m.size() << std::endl;
    
    return m;
}

int main(int argc, char *argv[])
{
    if (argc != 5 && (argc-2) % 3 != 0)  {
        std::cout << "usage: " << argv[0] << " <params.json> <tiffxyz> <winding> <weight> ..." << std::endl;
        std::cout << "   multiple triplets of <tiffxyz> <winding> <weight> can be used for a joint optimization" << std::endl;
        return EXIT_SUCCESS;
    }
    
    std::vector<QuadSurface*> surfs;
    std::vector<cv::Mat_<cv::Vec3f>> surf_points;
    std::vector<cv::Mat_<float>> winds;
    std::vector<cv::Mat_<uint8_t>> supports;
    std::vector<float> weights;
    std::vector<cv::Mat_<cv::Vec2d>> surf_locs;

    std::ifstream params_f(argv[1]);
    json params = json::parse(params_f);
    
    trace_mul = params.value("trace_mul", 1);
    dist_w = params.value("dist_w", 0.3);
    straight_w = params.value("straight_w", 0.02);
    surf_w = params.value("surf_w", 0.1);
    z_loc_loss_w = params.value("z_loc_loss_w", 0.0005);
    wind_w = params.value("wind_w", 100.0);
    wind_th = params.value("wind_th", 0.3);
    inpaint_back_range = params.value("inpaint_back_range", 40);
    int opt_w_short = params.value("opt_w", 4);
    
    int opt_w = opt_w_short;
    int large_opt_w = 32;
    int large_opt_every = 8;
    
    for(int n=0;n<argc/3;n++) {
        QuadSurface *surf = load_quad_from_tifxyz(argv[n*3+2]);
        
        cv::Mat_<float> wind = cv::imread(argv[n*3+3], cv::IMREAD_UNCHANGED);
                    
        cv::Mat_<cv::Vec3f> points = surf->rawPoints();
        
        for(int j=0;j<wind.rows;j++)
            for(int i=0;i<wind.cols;i++)
                if (points(j,i)[0] == -1)
                    wind(j,i) = NAN;
        
        surfs.push_back(surf);
        winds.push_back(wind);
        surf_points.push_back(points);
        weights.push_back(atof(argv[n*3+4]));
    }
    
    if (surfs.size() > 1) {
        for(int i=1;i<surfs.size();i++) {
            //try to find random matches between the surfaces, always coming from surf 0 for now
            std::vector<float> offsets;
            std::vector<float> offsets_rev;
            
            for(int r=0;r<1000;r++) {
                cv::Vec2i p = {rand() % surf_points[0].rows, rand() % surf_points[0].cols};
                if (surf_points[0](p)[0] == -1)
                    continue;
                
                
                cv::Vec3f ptr = surfs[i]->pointer();
                float res = surfs[i]->pointTo(ptr, surf_points[0](p), 2.0);
                
                if (res < 0 || res >= 2)
                    continue;
                
                cv::Vec3f loc = surfs[i]->loc_raw(ptr);

                offsets.push_back(winds[0](p) - at_int(winds[i], {loc[0],loc[1]}));
                offsets_rev.push_back(winds[0](p) + at_int(winds[i], {loc[0],loc[1]}));
            }
            
            std::sort(offsets.begin(), offsets.end());
            std::sort(offsets_rev.begin(), offsets_rev.end());
            std::cout << "off 0.1 " << offsets[offsets.size()*0.1] << std::endl;
            std::cout << "off 0.5 " << offsets[offsets.size()*0.5] << std::endl;
            std::cout << "off 0.9 " << offsets[offsets.size()*0.9] << std::endl;
            float div_fw = std::abs(offsets[offsets.size()*0.9] - offsets[offsets.size()*0.1]);
            
            
            std::cout << "off_rev 0.1 " << offsets_rev[offsets_rev.size()*0.1] << std::endl;
            std::cout << "off_rev 0.5 " << offsets_rev[offsets_rev.size()*0.5] << std::endl;
            std::cout << "off_rev 0.9 " << offsets_rev[offsets_rev.size()*0.9] << std::endl;
            float div_bw = std::abs(offsets_rev[offsets.size()*0.9] - offsets_rev[offsets.size()*0.1]);
            
            
            if (div_fw < div_bw)
                winds[i] += offsets[offsets.size()*0.5];
            else
                winds[i] = -winds[i] + offsets_rev[offsets_rev.size()*0.5];
        }
    }
    
    //safety margin so we don't access out of mat points
    int margin = 4*trace_mul;
    int pad_amount = margin+20;
    
    for(int s=0;s<surf_points.size();s++) {
        surf_points[s] = pad(surf_points[s], pad_amount, {-1,-1,-1});
        winds[s] = pad(winds[s], pad_amount, NAN);
        supports.push_back(cv::Mat_<uint8_t>(winds[0].size(), 0));
        surf_locs.push_back(cv::Mat_<cv::Vec2d>(winds[0].size(), {-1,-1}));
    }
    
    cv::Mat_<cv::Vec3f> points_in = surf_points[0];
    cv::Mat_<float> winding_in = winds[0].clone();

    float _min_w = 1000000;
    float _max_w = -1000000;
    
        
    for(int s=0;s<surf_points.size();s++)
        for(int j=0;j<surf_points[s].rows;j++)
            for(int i=0;i<surf_points[s].cols;i++)
                if (!std::isnan(winds[s](j,i))) {
                    _min_w = std::min(_min_w, winds[s](j,i));
                    _max_w = std::max(_max_w, winds[s](j,i));
                }
                
    //FIXME remove hardcoded limit on Z
    float mul_z = 1000.0/15000;
    
    int num_winds = std::ceil(_max_w) - std::floor(_min_w);
    
    int nw_step = 50;
    
    cv::Mat_<cv::Vec3f> normals(cv::Size(nw_step*num_winds, 1000), 0);
    cv::Mat_<float> normals_w(cv::Size(nw_step*num_winds,1000), 0);
    for(int s=0;s<1;s++) { //TODO winding number is too inaccurate - we could use the normals to refine it (but for now we just use the first trace)
        normals.setTo(0);
        normals_w.setTo(0);
        for(int j=0;j<surf_points[s].rows;j++)
            for(int i=0;i<surf_points[s].cols;i++) {
                if (surf_points[s](j, i)[0] == -1)
                    continue;
                cv::Vec3f n = grid_normal(surf_points[s], {static_cast<float>(i),static_cast<float>(j),0});
                cv::Vec3f p = surf_points[s](j,i);
                if (std::isnan(n[0]))
                    continue;
                
                int zi = p[2]*mul_z;
                int wi = (winds[s](j,i)-_min_w)/num_winds*normals.cols;
                normals(zi, wi) += n;
                normals_w(zi, wi) ++;
            }
            
//             for(int j=0;j<normals.rows;j++)
//                 for(int i=0;i<normals.cols;i++)
//                     if (normals_w(j,i) > 0) {
//                         normals(j,i) /= normals_w(j,i);
//                         normals_w(j,i) = 1;
//                     }
//             
//         cv::Mat_<cv::Vec3b> normcol;
//         normals.convertTo(normcol, CV_8U, 255);
//         cv::imwrite("init_normals"+std::to_string(s)+".tif", normcol);
    }
        
    for(int j=0;j<normals.rows;j++)
        for(int i=0;i<normals.cols;i++)
            if (normals_w(j,i) > 0) {
                normals(j,i) /= normals_w(j,i);
                normals_w(j,i) = 1;
            }
        
    cv::Mat_<cv::Vec3f> normals_in = normals.clone();
    cv::Mat_<float> normals_w_in = normals_w.clone();
    
    std::cout << "wind_step " << nw_step << std::endl;
    for(int r=0;r<50+num_winds;r++) {
        cv::Mat_<cv::Vec3f> normals_out(normals.size(), 0);
        cv::Mat_<float> normals_w_out(normals.size(), 0);
        
        if (r < 20) {
            for(int j=0;j<normals.rows;j++)
                for(int i=0;i<normals.cols-nw_step;i++) {
                    if (normals_w_in(j,i) && normals_w_in(j,i+nw_step)) {
                        cv::Vec3f avg = normals_in(j,i) + normals_in(j,i+nw_step);
                        avg *= 0.5;
                        normals_out(j,i) += avg;
                        normals_w_out(j,i) += 1;
                        normals_out(j,i+nw_step) += avg;
                        normals_w_out(j,i+nw_step) += 1;
                    }
                    else if (!normals_w_in(j,i) && normals_w_in(j,i+nw_step)) {
                        normals_out(j,i) += normals_in(j,i+nw_step);
                        normals_w_out(j,i) += 1;
                    }
                    else if (normals_w_in(j,i) && !normals_w_in(j,i+nw_step)) {
                        normals_out(j,i+nw_step) += normals_in(j,i);
                        normals_w_out(j,i+nw_step) += 1;
                    }
                }
            
            for(int j=0;j<normals.rows;j++)
                for(int i=0;i<normals.cols;i++)
                    if (normals_w(j,i)) {
                        normals_out(j,i) = normals(j,i);
                        normals_w_out(j,i) = 1;
                    }
                    else if (normals_w_out(j,i)) {
                        normals_out(j,i) /= normals_w_out(j,i);
                        normals_w_out(j,i) = 1;
                    }
                    
            normals_in = normals_out.clone();
            normals_w_in = normals_w_out.clone();
            normals_out.setTo(0);
            normals_w_out.setTo(0);
        }
        
        for(int j=0;j<normals.rows-1;j++)
            for(int i=0;i<normals.cols-1;i++)
                for (int oy=0;oy<=1;oy++)
                    for (int ox=0;ox<=1;ox++) {
                        if (ox == oy)
                            continue;
                        cv::Vec2i p1 = {j,i};
                        cv::Vec2i p2 = {j+oy,i+ox};
                        if (normals_w_in(p1) && normals_w_in(p2)) {
                            cv::Vec3f avg = normals_in(p1) + normals_in(p2);
                            avg *= 0.5;
                            normals_out(p1) += avg;
                            normals_w_out(p1) += 1;
                            normals_out(p2) += avg;
                            normals_w_out(p2) += 1;
                        }
                        else if (!normals_w_in(p1) && normals_w_in(p2)) {
                            normals_out(p1) += normals_in(p2);
                            normals_w_out(p1) += 1;
                        }
                        else if (normals_w_in(p1) && !normals_w_in(p2)) {
                            normals_out(p2) += normals_in(p1);
                            normals_w_out(p2) += 1;
                        }
                        
                        if (normals_w(p1)) {
                            normals_out(p1) = normals(p1);
                            normals_w_out(p1) = 1;
                        }
                    }
            
        for(int j=0;j<normals.rows;j++)
            for(int i=0;i<normals.cols;i++)
                if (normals_w(j,i)) {
                    normals_out(j,i) = normals(j,i);
                    normals_w_out(j,i) = 1;
                }
                else if (normals_w_out(j,i)) {
                    normals_out(j,i) /= normals_w_out(j,i);
                    normals_w_out(j,i) = 1;
                }
            
        normals_in = normals_out;
        normals_w_in = normals_w_out;
    }
    
    normals = normals_in;
    normals_w = normals_w_in;
    
    cv::GaussianBlur(normals,  normals, {3,3}, 0);

    cv::Mat_<cv::Vec3b> normcol;
    normals.convertTo(normcol, CV_8U, 255);
    cv::imwrite("normals.tif", normcol);
    
    int start_offset = 0;        
    
    cv::Rect bbox_src(std::max(margin,start_offset),margin,points_in.cols-std::max(margin,start_offset)-margin,points_in.rows-2*margin);
    
    float src_step = 20;
    float step = src_step*trace_mul;
    
    //needs to be large enough!
    int init_w = std::max(opt_w, margin);
    
    cv::Size size = {points_in.cols/trace_mul, points_in.rows/trace_mul};
    cv::Rect bbox_init = {bbox_src.x/trace_mul, bbox_src.y/trace_mul, bbox_src.width/trace_mul, bbox_src.height/trace_mul};
    cv::Rect bbox = {bbox_src.x/trace_mul, margin/trace_mul, bbox_src.width/trace_mul, (points_in.rows-2*margin)/trace_mul};

    cv::Mat_<uint8_t> state(size, 0);
    cv::Mat_<uint8_t> init_state(size, 0);
    cv::Mat_<cv::Vec3d> points(size, {-1,-1,-1});
    cv::Mat_<float> winding(size, NAN);
    cv::Mat_<float> init_errs(size, -1);
    
    cv::Mat_<uint8_t> fail_code(size, 0);
    
    cv::Rect first_col = {bbox_init.x,bbox_init.y,init_w,bbox_init.height};
    
    int last_miny, last_maxy;
    
    int col_first = first_col.height;
    int col_last = -1;
    for(int i=first_col.x;i<first_col.br().x;i++) {
        for(int j=first_col.y;j<first_col.br().y;j++) {            
            if (points_in(j*trace_mul,i*trace_mul)[0] != -1) {
                col_first = std::min(col_first, j);
                col_last = std::max(col_last, j);
            }
        }
    }
    
    std::cout << "init y range " << col_first << " " << col_last << std::endl;

    last_miny = col_first;
    last_maxy = col_last;
    
    for(int i=first_col.x;i<first_col.br().x;i++) {
        for(int j=col_first;j<=col_last;j++) {
            points(j,i) = points_in(j*trace_mul, i*trace_mul);
            winding(j,i) = winding_in(j*trace_mul, i*trace_mul);
            if (points(j,i)[0] != -1) {
                state(j,i) = STATE_LOC_VALID | STATE_COORD_VALID;
                surf_locs[0](j,i) = {static_cast<double>(j*trace_mul),static_cast<double>(i*trace_mul)};
            }
            else {
                if (points(j-1,i)[0] != -1) 
                    points(j, i) = points(j-1,i) + cv::Vec3d(0.1,0.1,0.1);
                else
                    points(j, i) = {static_cast<double>(rand()%1000),static_cast<double>(rand()%1000),static_cast<double>(rand()%1000)};
                state(j,i) = STATE_COORD_VALID;
            }
        }
    }
    
    cv::imwrite("state.tif", state*20);
    
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    // options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 10000;
    
    
    ceres::Solver::Options options_col;
    options_col.linear_solver_type = ceres::SPARSE_SCHUR;
#ifdef VC_USE_CUDA_SPARSE
    options_col.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;
#endif
    options_col.num_threads = 32;
    options_col.minimizer_progress_to_stdout = false;
    options_col.max_num_iterations = 10000;
    options_col.use_mixed_precision_solves = true;

    cv::Vec3f seed_coord = {-1,-1,-1};
    cv::Vec2i seed_loc = {-1,-1};
    
    std::vector<float> avg_wind(size.width);
    std::vector<int> wind_counts(size.width);
    std::vector<float> tgt_wind(size.width);
    {
        ceres::Problem problem_init;
        for(int j=first_col.y;j<first_col.br().y;j++)
            for(int i=first_col.x;i<first_col.br().x;i++) {
                create_centered_losses(problem_init, {j,i}, state, points_in, points, surf_locs[0], normals, -1, -1, step, LOSS_ON_SURF);
                

                if (loc_valid(state(j,i))) {
                    avg_wind[i] += winding_in(j*trace_mul,i*trace_mul);
                    wind_counts[i]++;

                    if (seed_loc[0] == -1) {
                        seed_loc = {j,i};
                        seed_coord = points_in(j*trace_mul,i*trace_mul);
                    }
                }
            }

        for(int i=first_col.x;i<first_col.br().x;i++) {
            std::cout << "init wind col " << i << " " << avg_wind[i] / wind_counts[i] << " " << wind_counts[i] << std::endl;
            avg_wind[i] /= wind_counts[i];
            wind_counts[i] = 1;
        }
        
        for(int i=bbox_init.x;i<bbox_init.br().x;i++)
            tgt_wind[i] = avg_wind[i];              
        
        for(int j=first_col.y;j<first_col.br().y;j++)
            for(int i=first_col.x;i<first_col.br().x;i++) {
                if (loc_valid(state(j,i))) {
                    problem_init.AddResidualBlock(Interp2DLoss<float>::Create(winding_in, avg_wind[i], wind_w), nullptr, &surf_locs[0](j,i)[0]);
                }
            }

        for(int j=first_col.y;j<first_col.br().y;j++)
            for(int i=first_col.x;i<first_col.br().x;i++) {
                cv::Vec2i p = {j,i};
                if (!loc_valid(state(p)) && coord_valid(state(p))) {
                    if (problem_init.HasParameterBlock(&surf_locs[0](p)[0]))
                        problem_init.SetParameterBlockVariable(&surf_locs[0](p)[0]);
                    if (problem_init.HasParameterBlock(&points(p)[0]))
                        problem_init.SetParameterBlockVariable(&points(p)[0]);
                }
                else {
                    if (problem_init.HasParameterBlock(&surf_locs[0](p)[0]))
                        problem_init.SetParameterBlockConstant(&surf_locs[0](p)[0]);
                    if (problem_init.HasParameterBlock(&points(p)[0]))
                        problem_init.SetParameterBlockConstant(&points(p)[0]);
                }
                    
            }
        
        ceres::Solver::Summary summary;
        ceres::Solve(options_col, &problem_init, &summary);
        std::cout << summary.BriefReport() << std::endl;
        
        for(int j=first_col.y;j<first_col.br().y;j++) {
            for(int i=first_col.x;i<first_col.br().x;i++) {
                if (problem_init.HasParameterBlock(&surf_locs[0](j,i)[0]))
                    problem_init.SetParameterBlockVariable(&surf_locs[0](j,i)[0]);
                if (problem_init.HasParameterBlock(&points(j,i)[0]))
                    problem_init.SetParameterBlockVariable(&points(j,i)[0]);
                }
            }
            
        problem_init.SetParameterBlockConstant(&points(seed_loc)[0]);
        problem_init.SetParameterBlockConstant(&surf_locs[0](seed_loc)[0]);
        
        ceres::Solve(options_col, &problem_init, &summary);
        std::cout << summary.BriefReport() << std::endl;
        
    }
    
    int expand_rate = 2;
    int shrink_inv_rate = 2;
    int dilate = 20;
    
    int lower_bound = last_miny;
    int upper_bound = last_maxy;
    int lower_loc = lower_bound;
    int upper_loc = upper_bound;
    
    std::vector<cv::Vec2i> neighs = {{0,-1},{-1,-1},{1,-1},{-2,-1},{2,-1}};
    
    cv::Mat_<float> surf_dist(points.size(), 0);
    
    for(int i=bbox.x+init_w;i<bbox.br().x;i++) {
        if (i % large_opt_every == 0 && i-bbox.x > large_opt_w)
            opt_w = large_opt_w;
        else
            opt_w = opt_w_short;
        
        tgt_wind[i] = 2*avg_wind[i-1]-avg_wind[i-2];
        
        float tgt_wind_x_i = (tgt_wind[i]-_min_w)/num_winds*normals.cols;
        
        std::cout << "wind tgt: " << tgt_wind[i] << " " << avg_wind[i-1] << " " << avg_wind[i-2] << std::endl;
        
        if (lower_bound > lower_loc-dilate)
            lower_bound = std::max(lower_loc-dilate,lower_bound-expand_rate);
        else if (lower_bound < lower_loc-dilate)
            lower_bound = std::min(lower_loc-dilate,lower_bound + (i % shrink_inv_rate));
            
        if (upper_bound < upper_loc+dilate)
            upper_bound = std::min(upper_loc+dilate,upper_bound+expand_rate);
        else if (upper_bound > upper_loc+dilate)
            upper_bound = std::max(upper_loc+dilate,upper_bound - (i % shrink_inv_rate));
        
        std::cout << "proc col " << i << std::endl;
        ceres::Problem problem_col;
#pragma omp parallel for
        for(int j=std::max(bbox.y,lower_bound);j<std::min(bbox.br().y,upper_bound+1);j++) {
            cv::Vec2i p = {j,i};

            points(p) = {-1,-1, -1};
            state(p) = 0;
            
            //FIXME
            for(auto n : neighs) {
                if (!state(p) && coord_valid(state(p+n))) {
                    points(p) = points(p+n)+cv::Vec3d(((j+i)%10)*0.01, ((j+i+1)%10)*0.01,((j+i+2)%10)*0.01);
                    state(p) = STATE_COORD_VALID;
                    break;
                }
            }
            
            if (points(p)[0] == -1)
                continue;

            ceres::Solver::Summary summary;
            ceres::Problem problem;
            cv::Mat_<cv::Vec2d> dummy_;
            create_centered_losses_left_large(problem, p, state, points_in, points, dummy_, normals, tgt_wind_x_i, mul_z, step, 0);
            problem.AddResidualBlock(ZCoordLoss::Create(seed_coord[2] - (j-seed_loc[0])*step, z_loc_loss_w), nullptr, &points(p)[0]);            
            
            ceres::Solve(options, &problem, &summary);
            
            init_errs(p) = sqrt(summary.final_cost/summary.num_residual_blocks);
        }
        
        std::cout << "init col done " << std::endl;
        
        last_miny--;
        last_maxy++;

        std::vector<cv::Vec2i> add_ps;
        std::vector<int> add_idxs;
        
        //TODO re-use existing locs if there!
#pragma omp parallel for
        for(int j=bbox.y;j<bbox.br().y;j++) {
            for(int o=0;o<=opt_w;o++) {
                //only add in area where we also inpaint
                if (!coord_valid(state(j,i-o)))
                    continue;
                
                cv::Vec2i po = {j,i-o};
                for(int s=0;s<surf_points.size();s++)
                {
                    cv::Vec2f loc = {0,0};
                    float res = find_loc_wind(loc, tgt_wind[i-o], surf_points[s], winds[s], points(po), 1.0, false);
                    loc = {loc[1],loc[0]};
                    if (res >= 0 &&
                        cv::norm(at_int(surf_points[s], {loc[1],loc[0]}) - cv::Vec3f(points(po))) <= 100
                        && cv::norm(at_int(winds[s], {loc[1],loc[0]}) - tgt_wind[i-o]) <= wind_th)
#pragma omp critical
                        {
                            surf_locs[s](po) = loc;
                            add_idxs.push_back(s);
                            add_ps.push_back(po);
                            supports[s](po) = 1;
                        }
                    else
                        supports[s](po) = 0;
                }
                
            }
        }
        
        //adjust state according to support counts
        for(int j=bbox.y;j<bbox.br().y;j++)
            for(int o=0;o<=opt_w;o++) {
                cv::Vec2i po = {j,i-o};

                int sup_count = 0;
                for(int s=0;s<surf_points.size();s++)
                    sup_count += supports[s](po);
                if (sup_count)
                    state(po) = STATE_LOC_VALID | STATE_COORD_VALID;
                else
                    state(po) &= ~STATE_LOC_VALID;
            }
        
        for(int n=0;n<add_idxs.size();n++) {
            int idx = add_idxs[n];
            cv::Vec2i p = add_ps[n];
            problem_col.AddResidualBlock(SurfaceLossD::Create(surf_points[idx], surf_w*weights[idx]), nullptr, &points(p)[0], &surf_locs[idx](p)[0]);
            problem_col.AddResidualBlock(Interp2DLoss<float>::Create(winds[idx], tgt_wind[p[1]], wind_w*weights[idx]), nullptr, &surf_locs[idx](p)[0]);
        }
        
        for(int j=bbox.y;j<bbox.br().y;j++)
            for(int o=std::max(bbox.x,i-inpaint_back_range);o<=i;o++)
                if (coord_valid(state(j, o))) {
                    float zw = sqrt(float(inpaint_back_range-(i-o))/inpaint_back_range);
                    float tgt_wind_x_o = (tgt_wind[o]-_min_w)/num_winds*normals.cols;
                    cv::Mat_<cv::Vec2d> dummy_;
                    create_centered_losses(problem_col, {j, o}, state, points_in, points, dummy_, normals, tgt_wind_x_o, mul_z, step, LOSS_ON_NORMALS);
                    problem_col.AddResidualBlock(ZCoordLoss::Create(seed_coord[2] - (j-seed_loc[0])*step, zw*z_loc_loss_w), nullptr, &points(j,o)[0]);
                }
        
        for(int j=bbox.y;j<bbox.br().y;j++) {
            for(int o=std::max(bbox.x,i-inpaint_back_range);o<=i;o++) {
                if (problem_col.HasParameterBlock(&points(j, o)[0]))
                    problem_col.SetParameterBlockVariable(&points(j, o)[0]);
            }
        }
        
        for(int j=bbox.y;j<bbox.br().y;j++) {
            for(int o=std::max(bbox.x,i-inpaint_back_range-2);o<=i-inpaint_back_range;o++)
                if (loc_valid(state(j, o)))
                    if (problem_col.HasParameterBlock(&points(j,o)[0]))
                        problem_col.SetParameterBlockConstant(&points(j,o)[0]);
        }
        
        for(int j=bbox.y;j<bbox.br().y;j++) {
            for(int o=std::max(bbox.x,i-inpaint_back_range-2);o<=i;o++)
                if (loc_valid(state(j, o)))
                    if (problem_col.HasParameterBlock(&points(j,o)[0]))
                        problem_col.SetParameterBlockConstant(&points(j,o)[0]);
        }
            
        std::cout << "start solve with blocks " << problem_col.NumResidualBlocks() << std::endl;
        ceres::Solver::Summary summary;
        ceres::Solve(options_col, &problem_col, &summary);
        
        std::cout << summary.BriefReport() << std::endl;
        
        for(int x=i-opt_w+1;x<=i;x++)
            for(int j=bbox.y;j<bbox.br().y;j++) {
                cv::Vec2i p = {j,x};
                if (problem_col.HasParameterBlock(&points(p)[0]))
                    problem_col.SetParameterBlockVariable(&points(p)[0]);
            }
            
            
        ceres::Solve(options_col, &problem_col, &summary);
        
        std::cout << summary.BriefReport() << std::endl;

        bool stop = false;
        float min_w = 1000, max_w = -1000;
        lower_loc = bbox.br().x+1;
        upper_loc = -1;
        for(int x=std::max(i-opt_w,bbox.x);x<=i;x++) {
            avg_wind[x] = 0;
            wind_counts[x] = 0;
            for(int j=bbox.y;j<bbox.br().y;j++) {
                
                cv::Vec2i p = {j,x};
                for (int s=0;s<surf_points.size();s++) {
                    if (supports[s](p)) {
                        if (loc_valid(surf_points[s], surf_locs[s](p))) {
                            if (abs(at_int(winds[s], {static_cast<float>(surf_locs[s](p)[1]),static_cast<float>(surf_locs[s](p)[0])}) - tgt_wind[x]) <= wind_th) {
                                //FIXME check wind + support + loc avlid
                                float int_w = at_int(winds[s], {static_cast<float>(surf_locs[s](p)[1]),static_cast<float>(surf_locs[s](p)[0])});
                                avg_wind[x] += int_w;
                                wind_counts[x]++;
                                if (i == x) {
                                    upper_loc = std::max(upper_loc, j);
                                    lower_loc = std::min(lower_loc, j);
                                    
                                    min_w = std::min(int_w-tgt_wind[x], min_w);
                                    max_w = std::max(int_w-tgt_wind[x], max_w);
                                }
                                
                            }
                            else
                                std::cout << "wind th " << abs(at_int(winds[s], {static_cast<float>(surf_locs[s](p)[1]),static_cast<float>(surf_locs[s](p)[0])}) - tgt_wind[x]) << " " << at_int(winds[s], {static_cast<float>(surf_locs[s](p)[1]),static_cast<float>(surf_locs[s](p)[0])}) << tgt_wind[x] << " " << std::endl;
                        }
                        else
                        {
                            std::cout << "lost a point! " << p << " " << s << " " << surf_locs[s](p) << std::endl;
                        }
                    }
                }
            }
            if (wind_counts[x])
                avg_wind[x] /= wind_counts[x];
            else {
                stop = true;
            }
        }
            
        if (avg_wind[i-1] < avg_wind[i-2] - wind_th/2) {
            stop = true;
            std::cout << "stopping wind is wrong! " << avg_wind[i-2] << " " << avg_wind[i-1]  << std::endl;
        }
            
        if (i % 10 == 0 || !wind_counts[i] || i == bbox.br().x-1 || stop) {
            std::vector<cv::Mat> chs;
            cv::split(points, chs);
            cv::imwrite("newx.tif",chs[0](bbox));
            cv::imwrite("newz.tif",chs[2](bbox));
            cv::imwrite("winding_out.tif",winding(bbox)+3);
            cv::imwrite("state.tif",state*20);
            cv::imwrite("init_errs.tif",init_errs(bbox));
            
            for(int s=0;s<supports.size();s++)
                cv::imwrite("supports"+std::to_string(s)+".tif",supports[s](bbox)*255);
        }
            
        if (!wind_counts[i]) {
            std::cout << "stopping as zero valid locations found! " << i << std::endl;
            break;
        }
        
        if (stop)
            break;
        
        
        std::cout << "avg wind number for col " << i << " : " << avg_wind[i] << " ( tgt was " << tgt_wind[i] << " ) using #" << wind_counts[i]  << " spread " << min_w << " - " << max_w << std::endl;
        wind_counts[i] = 1;

    }
    
    {
        QuadSurface *surf_full = new QuadSurface(points(bbox), surfs[0]->_scale/trace_mul);
        std::filesystem::path tgt_dir = "./";
        surf_full->meta = new nlohmann::json;
        (*surf_full->meta)["vc_fill_quadmesh_params"] = params;
        std::string name_prefix = "fuse_fill_";
        std::string uuid = name_prefix + time_str();
        std::filesystem::path seg_dir = tgt_dir / uuid;
        std::cout << "saving " << seg_dir << std::endl;
        surf_full->save(seg_dir, uuid);
        
        cv::Mat_<float> winding_ideal(winding.size(), NAN);
        for(int i=0;i<tgt_wind.size();i++)
            winding_ideal(cv::Rect(i,0,1,winding.rows)).setTo(tgt_wind[i]);

        cv::imwrite(seg_dir/"winding_exact.tif",winding);
        cv::imwrite(seg_dir/"winding.tif",winding_ideal);
    }
    
    // {
    //     cv::Mat_<cv::Vec3f> points_hr = points_hr_grounding(state, tgt_wind, winding_in, points, points_in, trace_mul);
    //     QuadSurface *surf_hr = new QuadSurface(points_hr, surfs[0]->_scale);
    //     std::filesystem::path tgt_dir = "./";
    //     surf_hr->meta = new nlohmann::json;
    //     (*surf_hr->meta)["vc_fill_quadmesh_params"] = params;
    //     std::string name_prefix = "testing_fill_hr_";
    //     std::string uuid = name_prefix + time_str();
    //     std::filesystem::path seg_dir = tgt_dir / uuid;
    //     std::cout << "saving " << seg_dir << std::endl;
    //     surf_hr->save(seg_dir, uuid);
    // }

    for (auto sm : surfs) {
        delete sm;
    }
    
    return EXIT_SUCCESS;
}
