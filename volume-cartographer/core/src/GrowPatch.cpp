#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/SurfaceModeling.hpp"
#include "vc/core/util/SurfaceArea.hpp"
#include "vc/core/util/OMPThreadPointCollection.hpp"
#include "vc/core/util/LifeTime.hpp"
#include "vc/core/types/ChunkedTensor.hpp"
#include <nlohmann/json.hpp>

#include "vc/core/util/NormalGridVolume.hpp"
#include "vc/core/util/GridStore.hpp"
#include "vc/core/util/CostFunctions.hpp"
#include "vc/core/util/HashFunctions.hpp"
#include "vc/core/util/NormalGridVolume.hpp"

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(views, xview.hpp)

#include <iostream>
#include <cctype>
#include <random>
#include <optional>
#include <cstdlib>
#include <limits>

#include "vc/tracer/Tracer.hpp"
#include "vc/ui/VCCollection.hpp"

namespace { // Anonymous namespace for local helpers

std::optional<uint32_t> environment_seed()
{
    static const std::optional<uint32_t> cached = []() -> std::optional<uint32_t> {
        const char* env = std::getenv("VC_GROWPATCH_RNG_SEED");
        if (!env || *env == '\0') {
            return std::nullopt;
        }

        char* end = nullptr;
        const unsigned long long value = std::strtoull(env, &end, 10);
        if (!end || *end != '\0' || value > std::numeric_limits<uint32_t>::max()) {
            return std::nullopt;
        }
        return static_cast<uint32_t>(value);
    }();

    return cached;
}

std::mt19937& thread_rng()
{
    static thread_local std::mt19937 rng = [] {
        if (const auto seed = environment_seed()) {
            return std::mt19937(*seed);
        }
        return std::mt19937(std::random_device{}());
    }();
    return rng;
}

[[maybe_unused]] void set_random_perturbation_seed(uint32_t seed)
{
    thread_rng().seed(seed);
}

cv::Vec3d random_perturbation(double max_abs_offset = 0.05) {
    std::uniform_real_distribution<double> dist(-max_abs_offset, max_abs_offset);
    auto& rng = thread_rng();
    return {dist(rng), dist(rng), dist(rng)};
}

struct Vec2iLess {
    bool operator()(const cv::Vec2i& a, const cv::Vec2i& b) const {
        if (a[0] != b[0]) {
            return a[0] < b[0];
        }
        return a[1] < b[1];
    }
};

class PointCorrection {
public:
    struct CorrectionCollection {
        std::vector<cv::Vec3f> tgts_;
        std::vector<cv::Vec2f> grid_locs_;
    };

    PointCorrection() = default;
    PointCorrection(const VCCollection& corrections) {
        const auto& collections = corrections.getAllCollections();
        if (collections.empty()) return;

        for (const auto& pair : collections) {
            const auto& collection = pair.second;
            if (collection.points.empty()) continue;

            CorrectionCollection new_collection;
            std::vector<ColPoint> sorted_points;
            sorted_points.reserve(collection.points.size());
            for (const auto& point_pair : collection.points) {
                sorted_points.push_back(point_pair.second);
            }
            std::sort(sorted_points.begin(), sorted_points.end(), [](const auto& a, const auto& b) {
                return a.id < b.id;
            });

            new_collection.tgts_.reserve(sorted_points.size());
            for (const auto& col_point : sorted_points) {
                new_collection.tgts_.push_back(col_point.p);
            }
            collections_.push_back(new_collection);
        }

        is_valid_ = !collections_.empty();
    }

    void init(const cv::Mat_<cv::Vec3f> &points) {
        if (!is_valid_ || points.empty()) {
            is_valid_ = false;
            return;
        }

        QuadSurface tmp(points, {1.0f, 1.0f});
        
        for (auto& collection : collections_) {
            cv::Vec3f ptr = tmp.pointer();

            // Initialize anchor point (lowest ID)
            float d = tmp.pointTo(ptr, collection.tgts_[0], 1.0f);
            cv::Vec3f loc_3d = tmp.loc_raw(ptr);
            std::cout << "base diff: " << d << loc_3d << std::endl;
            cv::Vec2f loc(loc_3d[0], loc_3d[1]);
            collection.grid_locs_.push_back({loc[0], loc[1]});

            // Initialize other points
            for (size_t i = 1; i < collection.tgts_.size(); ++i) {
                d = tmp.pointTo(ptr, collection.tgts_[i], 100.0f, 0);
                loc_3d = tmp.loc_raw(ptr);
                std::cout << "point diff: " << d << loc_3d << std::endl;
                loc = {loc_3d[0], loc_3d[1]};
                collection.grid_locs_.push_back({loc[0], loc[1]});
            }
        }
    }

    bool isValid() const { return is_valid_; }
    
    const std::vector<CorrectionCollection>& collections() const { return collections_; }

    std::vector<cv::Vec3f> all_tgts() const {
        std::vector<cv::Vec3f> flat_tgts;
        for (const auto& collection : collections_) {
            flat_tgts.insert(flat_tgts.end(), collection.tgts_.begin(), collection.tgts_.end());
        }
        return flat_tgts;
    }

    std::vector<cv::Vec2f> all_grid_locs() const {
        std::vector<cv::Vec2f> flat_locs;
        for (const auto& collection : collections_) {
            flat_locs.insert(flat_locs.end(), collection.grid_locs_.begin(), collection.grid_locs_.end());
        }
        return flat_locs;
    }

private:
    bool is_valid_ = false;
    std::vector<CorrectionCollection> collections_;
};

struct TraceData {
    TraceData(const std::vector<DirectionField> &direction_fields) : direction_fields(direction_fields) {};
    PointCorrection point_correction;
    const vc::core::util::NormalGridVolume *ngv = nullptr;
    const std::vector<DirectionField> &direction_fields;
};

struct TraceParameters {
    cv::Mat_<uint8_t> state;
    cv::Mat_<cv::Vec3d> dpoints;
    float unit;
};

enum LossType {
    DIST,
    STRAIGHT,
    DIRECTION,
    SNAP,
    NORMAL,
    COUNT
};

struct LossSettings {
    std::vector<float> w = std::vector<float>(LossType::COUNT, 0.0f);
    std::vector<cv::Mat_<float>> w_mats = std::vector<cv::Mat_<float>>(LossType::COUNT);

    LossSettings() {
        w[LossType::SNAP] = 0.1f;
        w[LossType::NORMAL] = 1.0f;
        w[LossType::STRAIGHT] = 0.2f;
        w[LossType::DIST] = 1.0f;
        w[LossType::DIRECTION] = 1.0f;
    }

    float operator()(LossType type, const cv::Vec2i& p) const {
        if (!w_mats[type].empty()) {
            return w_mats[type](p);
        }
        return w[type];
    }

    float& operator[](LossType type) {
        return w[type];
    }

    int z_min = -1;
    int z_max = std::numeric_limits<int>::max();
};

static std::vector<cv::Vec2i> parse_growth_directions(const nlohmann::json& params)
{
    static const std::vector<cv::Vec2i> kDefaultDirections = {
        {1, 0},   // down / +row
        {0, 1},   // right / +col
        {-1, 0},  // up / -row
        {0, -1}   // left / -col
    };

    const auto it = params.find("growth_directions");
    if (it == params.end()) {
        return kDefaultDirections;
    }

    const nlohmann::json& directions = *it;
    if (!directions.is_array()) {
        std::cerr << "growth_directions parameter must be an array of strings" << std::endl;
        return kDefaultDirections;
    }

    bool allow_down = false;
    bool allow_right = false;
    bool allow_up = false;
    bool allow_left = false;
    bool any_valid = false;

    for (const auto& entry : directions) {
        if (!entry.is_string()) {
            std::cerr << "Ignoring non-string entry in growth_directions" << std::endl;
            continue;
        }

        const std::string value = entry.get<std::string>();
        std::string lower;
        lower.reserve(value.size());
        for (char ch : value) {
            lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }

        if (lower == "all") {
            return kDefaultDirections;
        }
        if (lower == "down") {
            allow_down = true;
            any_valid = true;
            continue;
        }
        if (lower == "right") {
            allow_right = true;
            any_valid = true;
            continue;
        }
        if (lower == "up") {
            allow_up = true;
            any_valid = true;
            continue;
        }
        if (lower == "left") {
            allow_left = true;
            any_valid = true;
            continue;
        }

        std::cerr << "Unknown growth direction '" << value << "' ignored" << std::endl;
    }

    if (!any_valid) {
        return kDefaultDirections;
    }

    std::vector<cv::Vec2i> custom;
    custom.reserve(4);
    if (allow_down) custom.emplace_back(1, 0);
    if (allow_right) custom.emplace_back(0, 1);
    if (allow_up) custom.emplace_back(-1, 0);
    if (allow_left) custom.emplace_back(0, -1);

    if (custom.empty()) {
        return kDefaultDirections;
    }

    return custom;
}

} // namespace

// global CUDA to allow use to set to false globally
// in the case they have cuda avail, but do not want to use it
static bool g_use_cuda = true;

// Expose a simple toggle for CUDA usage so tools can honor JSON settings
void set_space_tracing_use_cuda(bool enable) {
    g_use_cuda = enable;
}

static int gen_straight_loss(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &o1, const cv::Vec2i &o2, const cv::Vec2i &o3, TraceParameters &params, const LossSettings &settings);
static int gen_normal_loss(ceres::Problem &problem, const cv::Vec2i &p, TraceParameters &params, const TraceData &trace_data, const LossSettings &settings);
static int conditional_normal_loss(int bit, const cv::Vec2i &p, cv::Mat_<uint16_t> &loss_status, ceres::Problem &problem, TraceParameters &params, const TraceData &trace_data, const LossSettings &settings);
static int gen_dist_loss(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &off, TraceParameters &params, const LossSettings &settings);

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
    const cv::Vec2i &o3, TraceParameters &params, const LossSettings &settings)
{
    if (!coord_valid(params.state(p+o1)))
        return 0;
    if (!coord_valid(params.state(p+o2)))
        return 0;
    if (!coord_valid(params.state(p+o3)))
        return 0;

    problem.AddResidualBlock(StraightLoss::Create(settings(LossType::STRAIGHT, p)), nullptr, &params.dpoints(p+o1)[0], &params.dpoints(p+o2)[0], &params.dpoints(p+o3)[0]);

    return 1;
}

static int gen_dist_loss(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &off, TraceParameters &params,
    const LossSettings &settings)
{
    // Add a loss saying that dpoints(p) and dpoints(p+off) should themselves be distance |off| apart
    // Here dpoints is a 2D grid mapping surface-space points to 3D volume space
    // So this says that distances should be preserved from volume to surface

    if (!coord_valid(params.state(p)))
        return 0;
    if (!coord_valid(params.state(p+off)))
        return 0;

    if (params.dpoints(p)[0] == -1)
        throw std::runtime_error("invalid loc passed as valid!");

    problem.AddResidualBlock(DistLoss::Create(params.unit*cv::norm(off),settings(LossType::DIST, p)), nullptr, &params.dpoints(p)[0], &params.dpoints(p+off)[0]);

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
    ceres::Problem &problem, TraceParameters &params, const LossSettings &settings)
{
    int set = 0;
    if (!loss_mask(bit, p, off, loss_status))
        set = set_loss_mask(bit, p, off, loss_status, gen_dist_loss(problem, p, off, params, settings));
    return set;
};

static int conditional_straight_loss(int bit, const cv::Vec2i &p, const cv::Vec2i &o1, const cv::Vec2i &o2, const cv::Vec2i &o3,
    cv::Mat_<uint16_t> &loss_status, ceres::Problem &problem, TraceParameters &params, const LossSettings &settings)
{
    int set = 0;
    if (!loss_mask(bit, p, o2, loss_status))
        set += set_loss_mask(bit, p, o2, loss_status, gen_straight_loss(problem, p, o1, o2, o3, params, settings));
    return set;
};

static int gen_normal_loss(ceres::Problem &problem, const cv::Vec2i &p, TraceParameters &params, const TraceData &trace_data, const LossSettings &settings)
{
    if (!trace_data.ngv) return 0;

    cv::Vec2i p_br = p + cv::Vec2i(1,1);
    if (!coord_valid(params.state(p)) || !coord_valid(params.state(p[0], p_br[1])) || !coord_valid(params.state(p_br[0], p[1])) || !coord_valid(params.state(p_br))) {
        return 0;
    }

    cv::Vec2i p_tr = {p[0], p[1] + 1};
    cv::Vec2i p_bl = {p[0] + 1, p[1]};

    // Points for the quad: A, B1, B2, C
    double* pA = &params.dpoints(p)[0];
    double* pB1 = &params.dpoints(p_tr)[0];
    double* pB2 = &params.dpoints(p_bl)[0];
    double* pC = &params.dpoints(p_br)[0];

    int count = 0;
    // int i = 1;
    for (int i = 0; i < 3; ++i) { // For each plane
        // bool direction_aware = (i == 0); // XY plane
        bool direction_aware = false; // this is not that simple ...
        // Loss with p as base point A
        problem.AddResidualBlock(NormalConstraintPlane::Create(*trace_data.ngv, i, settings(LossType::NORMAL, p), settings(LossType::SNAP, p), direction_aware, settings.z_min, settings.z_max), nullptr, pA, pB1, pB2, pC);
        // Loss with p_br as base point A
        problem.AddResidualBlock(NormalConstraintPlane::Create(*trace_data.ngv, i, settings(LossType::NORMAL, p), settings(LossType::SNAP, p), direction_aware, settings.z_min, settings.z_max), nullptr, pC, pB2, pB1, pA);
        // Loss with p_tr as base point A
        problem.AddResidualBlock(NormalConstraintPlane::Create(*trace_data.ngv, i, settings(LossType::NORMAL, p), settings(LossType::SNAP, p), direction_aware, settings.z_min, settings.z_max), nullptr, pB1, pC, pA, pB2);
        // Loss with p_bl as base point A
        problem.AddResidualBlock(NormalConstraintPlane::Create(*trace_data.ngv, i, settings(LossType::NORMAL, p), settings(LossType::SNAP, p), direction_aware, settings.z_min, settings.z_max), nullptr, pB2, pA, pC, pB1);
        count += 4;
    }

    //FIXME make params constant if not optimize-all is set

    return count;
}

static int conditional_normal_loss(int bit, const cv::Vec2i &p, cv::Mat_<uint16_t> &loss_status,
    ceres::Problem &problem, TraceParameters &params, const TraceData &trace_data, const LossSettings &settings)
{
    if (!trace_data.ngv) return 0;
    int set = 0;
    if (!loss_mask(bit, p, {0,0}, loss_status))
        set = set_loss_mask(bit, p, {0,0}, loss_status, gen_normal_loss(problem, p, params, trace_data, settings));
    return set;
};


// static void freeze_inner_params(ceres::Problem &problem, int edge_dist, const cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &out,
//     cv::Mat_<cv::Vec2d> &loc, cv::Mat_<uint16_t> &loss_status, int inner_flags)
// {
//     cv::Mat_<float> dist(state.size());
//
//     edge_dist = std::min(edge_dist,254);
//
//
//     cv::Mat_<uint8_t> masked;
//     bitwise_and(state, cv::Scalar(inner_flags), masked);
//
//
//     cv::distanceTransform(masked, dist, cv::DIST_L1, cv::DIST_MASK_3);
//
//     for(int j=0;j<dist.rows;j++)
//         for(int i=0;i<dist.cols;i++) {
//             if (dist(j,i) >= edge_dist && !loss_mask(7, {j,i}, {0,0}, loss_status)) {
//                 if (problem.HasParameterBlock(&out(j,i)[0]))
//                     problem.SetParameterBlockConstant(&out(j,i)[0]);
//                 if (!loc.empty() && problem.HasParameterBlock(&loc(j,i)[0]))
//                     problem.SetParameterBlockConstant(&loc(j,i)[0]);
//                 // set_loss_mask(7, {j,i}, {0,0}, loss_status, 1);
//             }
//             if (dist(j,i) >= edge_dist+2 && !loss_mask(8, {j,i}, {0,0}, loss_status)) {
//                 if (problem.HasParameterBlock(&out(j,i)[0]))
//                     problem.RemoveParameterBlock(&out(j,i)[0]);
//                 if (!loc.empty() && problem.HasParameterBlock(&loc(j,i)[0]))
//                     problem.RemoveParameterBlock(&loc(j,i)[0]);
//                 // set_loss_mask(8, {j,i}, {0,0}, loss_status, 1);
//             }
//         }
// }


// struct DSReader
// {
//     z5::Dataset *ds;
//     float scale;
//     ChunkCache *cache;
// };



int gen_direction_loss(ceres::Problem &problem,
    const cv::Vec2i &p,
    const int off_dist,
    cv::Mat_<uint8_t> &state,
    cv::Mat_<cv::Vec3d> &loc,
    std::vector<DirectionField> const &direction_fields,
    const LossSettings& settings)
{
    // Add losses saying that the local basis vectors of the patch at loc(p) should match those of the given fields

    if (!loc_valid(state(p)))
        return 0;

    cv::Vec2i const p_off_horz{p[0], p[1] + off_dist};
    cv::Vec2i const p_off_vert{p[0] + off_dist, p[1]};

    const float baseWeight = settings(LossType::DIRECTION, p);

    int count = 0;
    for (const auto &field: direction_fields) {
        const float totalWeight = baseWeight * field.weight;
        if (totalWeight <= 0.0f) {
            continue;
        }
        if (field.direction == "horizontal") {
            if (!loc_valid(state(p_off_horz)))
                continue;
            problem.AddResidualBlock(FiberDirectionLoss::Create(*field.field_ptr, field.weight_ptr.get(), totalWeight), nullptr, &loc(p)[0], &loc(p_off_horz)[0]);
        } else if (field.direction == "vertical") {
            if (!loc_valid(state(p_off_vert)))
                continue;
            problem.AddResidualBlock(FiberDirectionLoss::Create(*field.field_ptr, field.weight_ptr.get(), totalWeight), nullptr, &loc(p)[0], &loc(p_off_vert)[0]);
        } else if (field.direction == "normal") {
            if (!loc_valid(state(p_off_horz)) || !loc_valid(state(p_off_vert)))
                continue;
            problem.AddResidualBlock(NormalDirectionLoss::Create(*field.field_ptr, field.weight_ptr.get(), totalWeight), nullptr, &loc(p)[0], &loc(p_off_horz)[0], &loc(p_off_vert)[0]);
        } else {
            assert(false);
        }
        ++count;
    }

    return count;
}

//create all valid losses for this point
// Forward declarations
static int gen_corr_loss(ceres::Problem &problem, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &loc, TraceParameters &trace_params);
static int conditional_corr_loss(int bit, const cv::Vec2i &p, cv::Mat_<uint16_t> &loss_status,
                                 ceres::Problem &problem, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &loc, TraceParameters &trace_params);

static int add_continuous_losses(ceres::Problem &problem, const cv::Vec2i &p, TraceParameters &params,
    const TraceData &trace_data, const LossSettings &settings)
{
    //generate losses for point p
    int count = 0;

    if (p[0] < 2 || p[1] < 2 || p[1] >= params.state.cols-2 || p[0] >= params.state.rows-2)
        throw std::runtime_error("point too close to problem border!");

    //horizontal
    count += gen_straight_loss(problem, p, {0,-2},{0,-1},{0,0}, params, settings);
    count += gen_straight_loss(problem, p, {0,-1},{0,0},{0,1}, params, settings);
    count += gen_straight_loss(problem, p, {0,0},{0,1},{0,2}, params, settings);

    //vertical
    count += gen_straight_loss(problem, p, {-2,0},{-1,0},{0,0}, params, settings);
    count += gen_straight_loss(problem, p, {-1,0},{0,0},{1,0}, params, settings);
    count += gen_straight_loss(problem, p, {0,0},{1,0},{2,0}, params, settings);


    //diag1
    count += gen_straight_loss(problem, p, {-2,-2},{-1,-1},{0,0}, params, settings);
    count += gen_straight_loss(problem, p, {-1,-1},{0,0},{1,1}, params, settings);
    count += gen_straight_loss(problem, p, {0,0},{1,1},{2,2}, params, settings);

    //diag2
    count += gen_straight_loss(problem, p, {-2,2},{-1,1},{0,0}, params, settings);
    count += gen_straight_loss(problem, p, {-1,1},{0,0},{1,-1}, params, settings);
    count += gen_straight_loss(problem, p, {0,0},{1,-1},{2,-2}, params, settings);

    //direct neighboars
    count += gen_dist_loss(problem, p, {0,-1}, params, settings);
    count += gen_dist_loss(problem, p, {0,1}, params, settings);
    count += gen_dist_loss(problem, p, {-1,0}, params, settings);
    count += gen_dist_loss(problem, p, {1,0}, params, settings);

    //diagonal neighbors
    count += gen_dist_loss(problem, p, {1,-1}, params, settings);
    count += gen_dist_loss(problem, p, {-1,1}, params, settings);
    count += gen_dist_loss(problem, p, {1,1}, params, settings);
    count += gen_dist_loss(problem, p, {-1,-1}, params, settings);

    //gridstore normals
    // count += gen_normal_loss(problem, p                   , params, trace_data, settings);
    // count += gen_normal_loss(problem, p + cv::Vec2i(-1,-1), params, trace_data, settings);
    // count += gen_normal_loss(problem, p + cv::Vec2i( 0,-1), params, trace_data, settings);
    // count += gen_normal_loss(problem, p + cv::Vec2i(-1, 0), params, trace_data, settings);

    return count;
}

static int conditional_direction_loss(int bit,
    const cv::Vec2i &p,
    const int u_off,
    cv::Mat_<uint16_t> &loss_status,
    ceres::Problem &problem,
    cv::Mat_<uint8_t> &state,
    cv::Mat_<cv::Vec3d> &loc,
    const LossSettings& settings,
    std::vector<DirectionField> const &direction_fields)
{
    if (!direction_fields.size())
        return 0;

    int set = 0;
    cv::Vec2i const off{0, u_off};
    if (!loss_mask(bit, p, off, loss_status))
        set = set_loss_mask(bit, p, off, loss_status, gen_direction_loss(problem, p, u_off, state, loc, direction_fields, settings));
    return set;
};

//create only missing losses so we can optimize the whole problem
static int gen_corr_loss(ceres::Problem &problem, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &dpoints, TraceData& trace_data)
{
    if (!trace_data.point_correction.isValid()) {
        return 0;
    }

    const auto& pc = trace_data.point_correction;

    const auto& all_grid_locs = pc.all_grid_locs();
    if (all_grid_locs.empty()) {
        return 0;
    }

    cv::Vec2i p_br = p + cv::Vec2i(1,1);
    if (!coord_valid(state(p)) || !coord_valid(state(p[0], p_br[1])) || !coord_valid(state(p_br[0], p[1])) || !coord_valid(state(p_br))) {
        return 0;
    }

    std::vector<cv::Vec3f> filtered_tgts;
    std::vector<cv::Vec2f> filtered_grid_locs;

    const auto& all_tgts = pc.all_tgts();
    cv::Vec2i quad_loc_int = {p[1], p[0]};

    for (size_t i = 0; i < all_tgts.size(); ++i) {
        const auto& grid_loc = all_grid_locs[i];
        float dx = grid_loc[0] - quad_loc_int[0];
        float dy = grid_loc[1] - quad_loc_int[1];
        if (dx * dx + dy * dy <= 4.0 * 4.0) {
            filtered_tgts.push_back(all_tgts[i]);
            filtered_grid_locs.push_back(all_grid_locs[i]);
        }
    }

    if (filtered_tgts.empty()) {
        return 0;
    }

    auto points_correction_loss = new PointsCorrectionLoss(filtered_tgts, filtered_grid_locs, quad_loc_int);
    auto cost_function = new ceres::DynamicAutoDiffCostFunction<PointsCorrectionLoss>(
        points_correction_loss
    );

    std::vector<double*> parameter_blocks;
    cost_function->AddParameterBlock(3);
    parameter_blocks.push_back(&dpoints(p)[0]);
    cost_function->AddParameterBlock(3);
    parameter_blocks.push_back(&dpoints(p + cv::Vec2i(0, 1))[0]);
    cost_function->AddParameterBlock(3);
    parameter_blocks.push_back(&dpoints(p + cv::Vec2i(1, 0))[0]);
    cost_function->AddParameterBlock(3);
    parameter_blocks.push_back(&dpoints(p + cv::Vec2i(1, 1))[0]);

    cost_function->SetNumResiduals(1);

    problem.AddResidualBlock(cost_function, nullptr, parameter_blocks);

    return 1;
}

static int conditional_corr_loss(int bit, const cv::Vec2i &p, cv::Mat_<uint16_t> &loss_status,
    ceres::Problem &problem, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &out, TraceData& trace_data)
{
    if (!trace_data.point_correction.isValid()) return 0;
    int set = 0;
    if (!loss_mask(bit, p, {0,0}, loss_status))
        set = set_loss_mask(bit, p, {0,0}, loss_status, gen_corr_loss(problem, p, state, out, trace_data));
    return set;
};

static int add_missing_losses(ceres::Problem &problem, cv::Mat_<uint16_t> &loss_status, const cv::Vec2i &p,
    TraceParameters &params, TraceData& trace_data,
    const LossSettings &settings)
{
    //generate losses for point p
    int count = 0;

    //horizontal
    count += conditional_straight_loss(0, p, {0,-2},{0,-1},{0,0}, loss_status, problem, params, settings);
    count += conditional_straight_loss(0, p, {0,-1},{0,0},{0,1}, loss_status, problem, params, settings);
    count += conditional_straight_loss(0, p, {0,0},{0,1},{0,2}, loss_status, problem, params, settings);

    //vertical
    count += conditional_straight_loss(1, p, {-2,0},{-1,0},{0,0}, loss_status, problem, params, settings);
    count += conditional_straight_loss(1, p, {-1,0},{0,0},{1,0}, loss_status, problem, params, settings);
    count += conditional_straight_loss(1, p, {0,0},{1,0},{2,0}, loss_status, problem, params, settings);

    //diag1
    count += conditional_straight_loss(0, p, {-2,-2},{-1,-1},{0,0}, loss_status, problem, params, settings);
    count += conditional_straight_loss(0, p, {-1,-1},{0,0},{1,1}, loss_status, problem, params, settings);
    count += conditional_straight_loss(0, p, {0,0},{1,1},{2,2}, loss_status, problem, params, settings);

    //diag2
    count += conditional_straight_loss(1, p, {-2,2},{-1,1},{0,0}, loss_status, problem, params, settings);
    count += conditional_straight_loss(1, p, {-1,1},{0,0},{1,-1}, loss_status, problem, params, settings);
    count += conditional_straight_loss(1, p, {0,0},{1,-1},{2,-2}, loss_status, problem, params, settings);

    //direct neighboars h
    count += conditional_dist_loss(2, p, {0,-1}, loss_status, problem, params, settings);
    count += conditional_dist_loss(2, p, {0,1}, loss_status, problem, params, settings);

    //direct neighbors v
    count += conditional_dist_loss(3, p, {-1,0}, loss_status, problem, params, settings);
    count += conditional_dist_loss(3, p, {1,0}, loss_status, problem, params, settings);

    //diagonal neighbors
    count += conditional_dist_loss(4, p, {1,-1}, loss_status, problem, params, settings);
    count += conditional_dist_loss(4, p, {-1,1}, loss_status, problem, params, settings);

    count += conditional_dist_loss(5, p, {1,1}, loss_status, problem, params, settings);
    count += conditional_dist_loss(5, p, {-1,-1}, loss_status, problem, params, settings);

    //normal field
    count += conditional_direction_loss(9, p, 1, loss_status, problem, params.state, params.dpoints, settings, trace_data.direction_fields);
    count += conditional_direction_loss(9, p, -1, loss_status, problem, params.state, params.dpoints, settings, trace_data.direction_fields);

    //gridstore normals
    count += conditional_normal_loss(10, p                   , loss_status, problem, params, trace_data, settings);
    count += conditional_normal_loss(10, p + cv::Vec2i(-1,-1), loss_status, problem, params, trace_data, settings);
    count += conditional_normal_loss(10, p + cv::Vec2i( 0,-1), loss_status, problem, params, trace_data, settings);
    count += conditional_normal_loss(10, p + cv::Vec2i(-1, 0), loss_status, problem, params, trace_data, settings);

    //snapping
    count += conditional_corr_loss(11, p,                    loss_status, problem, params.state, params.dpoints, trace_data);
    count += conditional_corr_loss(11, p + cv::Vec2i(-1,-1), loss_status, problem, params.state, params.dpoints, trace_data);
    count += conditional_corr_loss(11, p + cv::Vec2i( 0,-1), loss_status, problem, params.state, params.dpoints, trace_data);
    count += conditional_corr_loss(11, p + cv::Vec2i(-1, 0), loss_status, problem, params.state, params.dpoints, trace_data);

    return count;
}


//optimize within a radius, setting edge points to constant
static float local_optimization(int radius, const cv::Vec2i &p, TraceParameters &params,
    TraceData& trace_data, LossSettings &settings, bool quiet = false, bool parallel = false)
{
    // This Ceres problem is parameterised by locs; residuals are progressively added as the patch grows enforcing that
    // all points in the patch are correct distance in 2D vs 3D space, not too high curvature, near surface prediction, etc.
    ceres::Problem problem;
    cv::Mat_<uint16_t> loss_status(params.state.size());

    int r_outer = radius+3;

    for(int oy=std::max(p[0]-r_outer,0);oy<=std::min(p[0]+r_outer,params.dpoints.rows-1);oy++)
        for(int ox=std::max(p[1]-r_outer,0);ox<=std::min(p[1]+r_outer,params.dpoints.cols-1);ox++)
            loss_status(oy,ox) = 0;

    for(int oy=std::max(p[0]-radius,0);oy<=std::min(p[0]+radius,params.dpoints.rows-1);oy++)
        for(int ox=std::max(p[1]-radius,0);ox<=std::min(p[1]+radius,params.dpoints.cols-1);ox++) {
            cv::Vec2i op = {oy, ox};
            if (cv::norm(p-op) <= radius) {
                add_missing_losses(problem, loss_status, op, params, trace_data, settings);
            }
        }
    for(int oy=std::max(p[0]-r_outer,0);oy<=std::min(p[0]+r_outer,params.dpoints.rows-1);oy++)
        for(int ox=std::max(p[1]-r_outer,0);ox<=std::min(p[1]+r_outer,params.dpoints.cols-1);ox++) {
            cv::Vec2i op = {oy, ox};
            if (cv::norm(p-op) > radius && problem.HasParameterBlock(&params.dpoints(op)[0]))
                problem.SetParameterBlockConstant(&params.dpoints(op)[0]);
        }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 1000;

    // options.function_tolerance = 1e-4;
    // options.use_nonmonotonic_steps = true;
    // options.use_mixed_precision_solves = true;
    // options.max_num_refinement_iterations = 3;
    // options.use_inner_iterations = true;


    if (parallel)
        options.num_threads = omp_get_max_threads();

//    if (problem.NumParameterBlocks() > 1) {
//        options.use_inner_iterations = true;
//    }
// // NOTE currently CPU seems always faster (40x , AMD 5800H vs RTX 3080 mobile, even a 5090 would probably be slower?)
// #ifdef VC_USE_CUDA_SPARSE
//     // Check if Ceres was actually built with CUDA sparse support
//     if (g_use_cuda) {
//         if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::CUDA_SPARSE)) {
//             options.linear_solver_type = ceres::SPARSE_SCHUR;
//             // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
//             options.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;
//
//             // if (options.linear_solver_type == ceres::SPARSE_SCHUR) {
//                 options.use_mixed_precision_solves = true;
//             // }
//         } else {
//             std::cerr << "Warning: use_cuda=true but Ceres was not built with CUDA sparse support. Falling back to CPU sparse." << std::endl;
//         }
//     }
// #endif
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


QuadSurface *tracer(z5::Dataset *ds, float scale, ChunkCache *cache, cv::Vec3f origin, const nlohmann::json &params, const std::string &cache_root, float voxelsize, std::vector<DirectionField> const &direction_fields, QuadSurface* resume_surf, const std::filesystem::path& tgt_path, const nlohmann::json& meta_params, const VCCollection &corrections)
{
    TraceData trace_data(direction_fields);
    LossSettings loss_settings;
    TraceParameters trace_params;


    int stop_gen = params.value("generations", 100);
    float step = params.value("step_size", 20.0f);
    trace_params.unit = step*scale;
    int rewind_gen = params.value("rewind_gen", -1);
    loss_settings.z_min = params.value("z_min", -1);
    loss_settings.z_max = params.value("z_max", std::numeric_limits<int>::max());
    ALifeTime f_timer("empty space tracing\n");
    // DSReader reader = {ds,scale,cache};
    std::unique_ptr<vc::core::util::NormalGridVolume> ngv;
    if (params.contains("normal_grid_path")) {
        ngv = std::make_unique<vc::core::util::NormalGridVolume>(params["normal_grid_path"].get<std::string>());
        if (ngv->metadata()["spiral-step"] != step) {
            throw std::runtime_error("step_size parameter mismatch between normal grid volume and tracer.");
        }
    }
    trace_data.ngv = ngv.get();

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
    // float Ts = step*scale;

    // The following track the state of the patch; they are each as big as the largest possible patch but initially empty
    // - locs defines the patch! It says for each 2D position, which 3D position it corresponds to
    // - state tracks whether each 2D position is part of the patch yet, and whether its 3D position has been found
    trace_params.dpoints = cv::Mat_<cv::Vec3d>(size,cv::Vec3f(-1,-1,-1));
    trace_params.state = cv::Mat_<uint8_t>(size,0);
    cv::Mat_<uint16_t> generations(size, (uint16_t)0);
    cv::Mat_<uint8_t> phys_fail(size,0);
    // cv::Mat_<float> init_dist(size,0);
    cv::Mat_<uint16_t> loss_status(cv::Size(w,h),0);
    cv::Rect used_area;
    int generation = 1;
    int succ = 0;  // number of quads successfully added to the patch (each of size approx. step**2)

    auto create_surface_from_state = [&, &f_timer = *timer]() {
        cv::Rect used_area_safe = used_area;
        used_area_safe.x -= 2;
        used_area_safe.y -= 2;
        used_area_safe.width += 4;
        used_area_safe.height += 4;
        cv::Mat_<cv::Vec3d> points_crop = trace_params.dpoints(used_area_safe);
        cv::Mat_<uint16_t> generations_crop = generations(used_area_safe);

        auto surf = new QuadSurface(points_crop, {1/T, 1/T});
        surf->setChannel("generations", generations_crop);

        const double area_est_vx2 = vc::surface::computeSurfaceAreaVox2(*surf);
        const double voxel_size_d = static_cast<double>(voxelsize);
        const double area_est_cm2 = area_est_vx2 * voxel_size_d * voxel_size_d / 1e8;

        surf->meta = new nlohmann::json(meta_params);
        (*surf->meta)["area_vx2"] = area_est_vx2;
        (*surf->meta)["area_cm2"] = area_est_cm2;
        (*surf->meta)["max_gen"] = generation;
        (*surf->meta)["seed"] = {origin[0], origin[1], origin[2]};
        (*surf->meta)["elapsed_time_s"] = f_timer.seconds();

        return surf;
    };

    cv::Vec3f vx = {1,0,0};
    cv::Vec3f vy = {0,1,0};

    // ceres::Problem big_problem;
    int loss_count = 0;
    double last_elapsed_seconds = 0.0;
    int last_succ = 0;

    std::cout << "lets go! " << std::endl;

    if (resume_surf) {
        std::cout << "resuime! " << std::endl;
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

        int min_gen = std::numeric_limits<int>::max();
        x0 = -1;
        y0 = -1;
        for (int j = 0; j < resume_points.rows; ++j) {
            for (int i = 0; i < resume_points.cols; ++i) {
                int target_y = pad_y + j;
                int target_x = pad_x + i;
                uint16_t gen = resume_generations.at<uint16_t>(j, i);
                if (gen > 0 && gen <= start_gen && resume_points(j,i)[0] != -1) {
                    trace_params.dpoints(target_y, target_x) = resume_points(j, i);
                    generations(target_y, target_x) = gen;
                    succ++;
                    trace_params.state(target_y, target_x) = STATE_LOC_VALID | STATE_COORD_VALID;
                    if (gen < min_gen) {
                        min_gen = gen;
                        x0 = target_x;
                        y0 = target_y;
                    }
                }
            }
        }

        trace_data.point_correction = PointCorrection(corrections);

        if (trace_data.point_correction.isValid()) {
            trace_data.point_correction.init(trace_params.dpoints);

            std::cout << "Resuming with " << trace_data.point_correction.all_grid_locs().size() << " correction points." << std::endl;
            cv::Mat mask = resume_surf->channel("mask");
            if (!mask.empty()) {
                std::vector<std::vector<cv::Point2f>> all_hulls;
                for (const auto& collection : trace_data.point_correction.collections()) {
                    if (collection.grid_locs_.empty()) continue;

                    std::vector<cv::Point2f> points_for_hull;
                    points_for_hull.reserve(collection.grid_locs_.size());
                    for (const auto& loc : collection.grid_locs_) {
                        points_for_hull.emplace_back(loc[0], loc[1]);
                    }
                    
                    std::vector<cv::Point2f> hull_points;
                    cv::convexHull(points_for_hull, hull_points);
                    if (!hull_points.empty()) {
                        all_hulls.push_back(hull_points);
                    }
                }

                for (int j = 0; j < mask.rows; ++j) {
                    for (int i = 0; i < mask.cols; ++i) {
                        if (mask.at<uint8_t>(j, i) == 0 && trace_params.state(pad_y + j, pad_x + i)) {
                            int target_y = pad_y + j;
                            int target_x = pad_x + i;
                            cv::Point2f p(target_x, target_y);
                            bool keep = false;
                            for (const auto& hull : all_hulls) {
                                if (cv::pointPolygonTest(hull, p, false) >= 0) {
                                    keep = true;
                                    break;
                                }
                            }
                            if (!keep) {
                                trace_params.state(target_y, target_x) = 0;
                                trace_params.dpoints(target_y, target_x) = cv::Vec3d(-1,-1,-1);
                                generations(target_y, target_x) = 0;
                                succ--;
                            }
                        }
                    }
                }
            }

            struct OptCenter {
                cv::Vec2i center;
                int radius;
            };
            std::vector<OptCenter> opt_centers;

            for (const auto& collection : trace_data.point_correction.collections()) {
                if (collection.grid_locs_.empty()) continue;

                cv::Vec2f avg_loc(0,0);
                for (const auto& loc : collection.grid_locs_) {
                    avg_loc += loc;
                }
                avg_loc *= (1.0f / collection.grid_locs_.size());

                float max_dist = 0.0f;
                for (const auto& loc : collection.grid_locs_) {
                    max_dist = std::max(max_dist, (float)cv::norm(loc - avg_loc));
                }

                int radius = 8 + static_cast<int>(std::ceil(max_dist));
                cv::Vec2i corr_center_i = { (int)std::round(avg_loc[1]), (int)std::round(avg_loc[0]) };
                opt_centers.push_back({corr_center_i, radius});

                std::cout << "correction opt centered at " << avg_loc << " with radius " << radius << std::endl;
                LossSettings loss_inpaint = loss_settings;
                loss_inpaint[SNAP] = 0.0;
                loss_inpaint[DIST] *= 0.1;
                local_optimization(radius, corr_center_i, trace_params, trace_data, loss_inpaint, false, true);
                loss_inpaint[SNAP] = 0.01;
                local_optimization(radius, corr_center_i, trace_params, trace_data, loss_inpaint, false, true);
                loss_inpaint[SNAP] = 0.05;
                local_optimization(radius, corr_center_i, trace_params, trace_data, loss_inpaint, false, true);
                local_optimization(radius, corr_center_i, trace_params, trace_data, loss_settings, false, true);
                local_optimization(radius, corr_center_i, trace_params, trace_data, loss_settings, false, true);
            }

            trace_data.point_correction = PointCorrection();

            for (const auto& opt_params : opt_centers) {
                LossSettings loss_inpaint = loss_settings;
                loss_inpaint[DIST] *= 0.1;
                local_optimization(opt_params.radius, opt_params.center, trace_params, trace_data, loss_inpaint, false, true);
            }

        }

        // Rebuild fringe from valid points
        for (int j = used_area.y; j < used_area.br().y; ++j) {
            for (int i = used_area.x; i < used_area.br().x; ++i) {
                if (trace_params.state(j, i) & STATE_LOC_VALID) {
                    fringe.push_back({j, i});
                }
            }
        }

        last_succ = succ;
        last_elapsed_seconds = f_timer.seconds();
        std::cout << "Resuming from generation " << generation << " with " << fringe.size() << " points. Initial loss count: " << loss_count << std::endl;

    } else {
        // Initialise the trace at the center of the available area, as a tiny single-quad patch at the seed point
        used_area = cv::Rect(x0,y0,2,2);
        //these are locations in the local volume!
        trace_params.dpoints(y0,x0) = origin;
        trace_params.dpoints(y0,x0+1) = origin+vx*0.1;
        trace_params.dpoints(y0+1,x0) = origin+vy*0.1;
        trace_params.dpoints(y0+1,x0+1) = origin+vx*0.1 + vy*0.1;

        trace_params.state(y0,x0) = STATE_LOC_VALID | STATE_COORD_VALID;
        trace_params.state(y0+1,x0) = STATE_LOC_VALID | STATE_COORD_VALID;
        trace_params.state(y0,x0+1) = STATE_LOC_VALID | STATE_COORD_VALID;
        trace_params.state(y0+1,x0+1) = STATE_LOC_VALID | STATE_COORD_VALID;

        generations(y0,x0) = 1;
        generations(y0,x0+1) = 1;
        generations(y0+1,x0) = 1;
        generations(y0+1,x0+1) = 1;

        fringe.push_back({y0,x0});
        fringe.push_back({y0+1,x0});
        fringe.push_back({y0,x0+1});
        fringe.push_back({y0+1,x0+1});

        std::cout << "init loss count " << loss_count << std::endl;
    }

    int succ_start = succ;

    // Solve the initial optimisation problem, just placing the first four vertices around the seed
    ceres::Solver::Summary big_summary;
    //just continue on resume no additional global opt
    if (!resume_surf) {
        local_optimization(8, {y0,x0}, trace_params, trace_data, loss_settings, true);
    }

    // Prepare a new set of Ceres options used later during local solves
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 200;
    options.function_tolerance = 1e-3;


    auto neighs = parse_growth_directions(params);

    int local_opt_r = 3;

    std::cout << "lets start fringe: " << fringe.size() << std::endl;

    while (!fringe.empty()) {
        bool global_opt = generation <= 50 && !resume_surf;

        ALifeTime timer_gen;
        timer_gen.del_msg = "time per generation ";

        int phys_fail_count_gen = 0;
        generation++;
        if (stop_gen && generation >= stop_gen)
            break;

        // For every point in the fringe (where we might expand the patch outwards), add to cands all
        // new 2D points we might add to the patch (and later find the corresponding 3D point for)
        for(const auto& p : fringe)
        {
            for(const auto& n : neighs)
                if (bounds.contains(cv::Point(p+n))
                    && (trace_params.state(p+n) & STATE_PROCESSING) == 0
                    && (trace_params.state(p+n) & STATE_LOC_VALID) == 0) {
                    trace_params.state(p+n) |= STATE_PROCESSING;
                    cands.push_back(p+n);
                }
        }
        std::cout << "gen " << generation << " processing " << cands.size() << " fringe cands (total done " << succ << " fringe: " << fringe.size() << ")" << std::endl;
        fringe.resize(0);

        std::cout << "cands " << cands.size() << std::endl;

        int succ_gen = 0;
        std::vector<cv::Vec2i> succ_gen_ps;

        // Build a structure that allows parallel iteration over cands, while avoiding any two threads simultaneously
        // considering two points that are too close to each other...
        OmpThreadPointCol cands_threadcol(local_opt_r*2+1, cands);

        // ...then start iterating over candidates in parallel using the above to yield points
#pragma omp parallel
        {
            CachedChunked3dInterpolator<uint8_t,thresholdedDistance> interp(proc_tensor);
//             int idx = rand() % cands.size();
            while (true) {
                int r = 1;
                double phys_fail_th = 0.1;
                cv::Vec2i p = cands_threadcol.next();
                if (p[0] == -1)
                    break;

                if (trace_params.state(p) & (STATE_LOC_VALID | STATE_COORD_VALID))
                    continue;

                // p is now a 2D point we consider adding to the patch; find the best 3D point to map it to

                // Iterate all adjacent points that are in the patch, and find their 3D locations
                int ref_count = 0;
                cv::Vec3d avg = {0,0,0};
                std::vector<cv::Vec2i> srcs;
                for(int oy=std::max(p[0]-r,0);oy<=std::min(p[0]+r,trace_params.dpoints.rows-1);oy++)
                    for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,trace_params.dpoints.cols-1);ox++)
                        if (trace_params.state(oy,ox) & STATE_LOC_VALID) {
                            ref_count++;
                            avg += trace_params.dpoints(oy,ox);
                            srcs.push_back({oy,ox});
                        }

                // Of those adjacent points, find the one that itself has most adjacent in-patch points
                cv::Vec2i best_l = srcs[0];
                int best_ref_l = -1;
                int rec_ref_sum = 0;
                for(cv::Vec2i l : srcs) {
                    int ref_l = 0;
                    for(int oy=std::max(l[0]-r,0);oy<=std::min(l[0]+r,trace_params.dpoints.rows-1);oy++)
                        for(int ox=std::max(l[1]-r,0);ox<=std::min(l[1]+r,trace_params.dpoints.cols-1);ox++)
                            if (trace_params.state(oy,ox) & STATE_LOC_VALID)
                                ref_l++;

                    rec_ref_sum += ref_l;

                    if (ref_l > best_ref_l) {
                        best_l = l;
                        best_ref_l = ref_l;
                    }
                }

                // Initial guess for the corresponding 3D location is a perturbation of the position of the best-connected neighbor
                avg /= ref_count;

                //"fast" skip based on avg z value out of limits
                if (avg[2] < loss_settings.z_min || avg[2] > loss_settings.z_max)
                    continue;

                cv::Vec3d init = trace_params.dpoints(best_l) + random_perturbation();
                trace_params.dpoints(p) = init;

                // Set up a new local optimzation problem for the candidate point and its neighbors (initially just distance
                // and curvature losses, not nearness-to-surface)
                ceres::Problem problem;

                trace_params.state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
                int local_loss_count = add_continuous_losses(problem, p, trace_params, trace_data, loss_settings);

                std::vector<double*> parameter_blocks;
                problem.GetParameterBlocks(&parameter_blocks);
                for (auto& block : parameter_blocks) {
                    problem.SetParameterBlockConstant(block);
                }
                problem.SetParameterBlockVariable(&trace_params.dpoints(p)[0]);

                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);

                generations(p) = generation;

#pragma omp critical
                {
                    succ++;
                    succ_gen++;
                    if (!used_area.contains(cv::Point(p[1],p[0]))) {
                        used_area = used_area | cv::Rect(p[1],p[0],1,1);
                    }

                    fringe.push_back(p);
                    succ_gen_ps.push_back(p);
                }

                for (int i=1;i<local_opt_r;i++)
                    local_optimization(i, p, trace_params, trace_data, loss_settings, true);
            }  // end parallel iteration over cands
        }

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

                    local_optimization(8, p, trace_params, trace_data, loss_settings, true);
                }
            }
        }
        else {
            //we do the global opt only every 8 gens, as every add does a small local solve anyweays
            if (generation % 8 == 0) {
                local_optimization(stop_gen+10, {y0,x0}, trace_params, trace_data, loss_settings, false, true);
            }
        }

        cands.resize(0);

        // --- Speed Reporting ---
        double elapsed_seconds = f_timer.seconds();
        double seconds_this_gen = elapsed_seconds - last_elapsed_seconds;
        int succ_this_gen = succ - last_succ;

        double const vx_per_quad = (double)step * step;
        double const voxelsize_mm = (double)voxelsize / 1000.0;
        double const voxelsize_m = (double)voxelsize / 1000000.0;
        double const mm2_per_quad = vx_per_quad * voxelsize_mm * voxelsize_mm;
        double const m2_per_quad = vx_per_quad * voxelsize_m * voxelsize_m;

        double const total_area_mm2 = succ * mm2_per_quad;
        double const total_area_m2 = succ * m2_per_quad;

        double const total_area_mm2_run = (succ-succ_start) * mm2_per_quad;
        double const total_area_m2_run = (succ-succ_start) * m2_per_quad;

        double avg_speed_mm2_s = (elapsed_seconds > 0) ? (total_area_mm2_run / elapsed_seconds) : 0.0;
        double current_speed_mm2_s = (seconds_this_gen > 0) ? (succ_this_gen * mm2_per_quad / seconds_this_gen) : 0.0;
        double avg_speed_m2_day = (elapsed_seconds > 0) ? (total_area_m2_run / (elapsed_seconds / (24.0 * 3600.0))) : 0.0;

        printf("-> done %d | fringe %ld | area %.2f mm^2 (%.6f m^2) | avg speed %.2f mm^2/s (%.6f m^2/day) | current speed %.2f mm^2/s\n",
               succ, (long)fringe.size(), total_area_mm2, total_area_m2, avg_speed_mm2_s, avg_speed_m2_day, current_speed_mm2_s);

        last_elapsed_seconds = elapsed_seconds;
        last_succ = succ;

        timer_gen.unit = succ_gen * vx_per_quad;
        timer_gen.unit_string = "vx^2";
        // print_accessor_stats();

        int snapshot_interval = params.value("snapshot-interval", 0);
        if (!tgt_path.empty() && snapshot_interval > 0 && generation % snapshot_interval == 0) {
            QuadSurface* surf = create_surface_from_state();
            surf->save(tgt_path, true);
            delete surf;
            std::cout << "saved snapshot in " << tgt_path << std::endl;
        }

    }  // end while fringe is non-empty
    delete timer;

    QuadSurface* surf = create_surface_from_state();
    const double area_est_vx2 = vc::surface::computeSurfaceAreaVox2(*surf);
    const double voxel_size_d = static_cast<double>(voxelsize);
    const double area_est_cm2 = area_est_vx2 * voxel_size_d * voxel_size_d / 1e8;
    printf("generated surface %f vx^2 (%f cm^2)\n", area_est_vx2, area_est_cm2);

    return surf;
}
