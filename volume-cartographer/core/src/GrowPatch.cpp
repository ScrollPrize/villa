#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
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

#include "z5/factory.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/dataset.hxx"

#include <xtensor/views/xview.hpp>
#include <xtensor/containers/xtensor.hpp>
#include "edt.hpp"

#include <iostream>
#include <cctype>
#include <random>
#include <optional>
#include <cstdlib>
#include <limits>
#include <memory>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <mutex>
#include <array>
#include <omp.h>  // ensure omp_get_max_threads() is declared

#include "vc/tracer/Tracer.hpp"
#include "vc/ui/VCCollection.hpp"
#include "vc/tracer/NeuralTracerConnection.h"

DirectionField::DirectionField(std::string dir,
                               std::unique_ptr<Chunked3dVec3fFromUint8> field,
                               std::unique_ptr<Chunked3dFloatFromUint8> weight_dataset,
                               float weight)
    : direction(std::move(dir))
    , field_ptr(std::move(field))
    , weight_ptr(std::move(weight_dataset))
    , weight(weight)
{
}
DirectionField::~DirectionField() = default;
DirectionField::DirectionField(DirectionField&&) noexcept = default;
DirectionField& DirectionField::operator=(DirectionField&&) noexcept = default;

#define LOSS_STRAIGHT 1
#define LOSS_DIST 2
#define LOSS_NORMALSNAP 4
#define LOSS_SDIR 8
#define LOSS_SPACELINE 16
#define LOSS_3DNORMALLINE 32

namespace { // Anonymous namespace for local helpers

struct SDTContext;

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

template <typename T>
static bool point_in_bounds(const cv::Mat_<T>& mat, const cv::Vec2i& p)
{
    return p[0] >= 0 && p[0] < mat.rows && p[1] >= 0 && p[1] < mat.cols;
}

static cv::Mat_<uchar> make_approved_mask(const cv::Mat& approval,
                                          const cv::Rect& resume_area,
                                          const cv::Size& trace_size)
{
    if (approval.empty()) {
        return {};
    }
    if (resume_area.width <= 0 || resume_area.height <= 0) {
        return {};
    }
    if (approval.rows != resume_area.height || approval.cols != resume_area.width) {
        std::cout << "cell reopt: approval mask size mismatch (approval "
                  << approval.cols << "x" << approval.rows
                  << " vs resume area " << resume_area.width << "x" << resume_area.height
                  << ")" << std::endl;
        return {};
    }

    cv::Mat_<uchar> approved(trace_size, static_cast<uchar>(1));
    const bool is_rgb = approval.channels() == 3;

    for (int r = 0; r < approval.rows; ++r) {
        for (int c = 0; c < approval.cols; ++c) {
            int tr = resume_area.y + r;
            int tc = resume_area.x + c;
            if (tr < 0 || tr >= approved.rows || tc < 0 || tc >= approved.cols) {
                continue;
            }
            bool is_approved = false;
            if (is_rgb) {
                const cv::Vec3b v = approval.at<cv::Vec3b>(r, c);
                is_approved = (v[0] != 0 || v[1] != 0 || v[2] != 0);
            } else {
                is_approved = approval.at<uint8_t>(r, c) != 0;
            }
            approved(tr, tc) = static_cast<uchar>(is_approved ? 1 : 0);
        }
    }

    return approved;
}

struct lineLossDistance
{
    enum {BORDER = 16};
    enum {CHUNK_SIZE = 64};
    enum {FILL_V = 0};

    explicit lineLossDistance(float threshold_in = 170.0f, bool invert_in = false)
        : threshold(threshold_in), invert(invert_in)
    {
        UNIQUE_ID_STRING = "spaceline_" + std::to_string(BORDER) + "_" + std::to_string(CHUNK_SIZE) + "_" +
                           std::to_string(static_cast<int>(std::round(threshold))) + "_" +
                           std::to_string(static_cast<int>(invert));
    }

    template <typename T, typename E>
    void compute(const T &large, T &small, const cv::Vec3i &offset_large)
    {
        (void)offset_large;
        const int s = CHUNK_SIZE + 2 * BORDER;
        const size_t voxels = static_cast<size_t>(s) * s * s;

        std::vector<uint8_t> binary(voxels, 1);
        const uint8_t thr = static_cast<uint8_t>(std::clamp(threshold, 0.0f, 255.0f));

#pragma omp parallel for
        for (int z = 0; z < s; ++z) {
            for (int y = 0; y < s; ++y) {
                for (int x = 0; x < s; ++x) {
                    const uint8_t v = large(z, y, x);
                    bool fg = v >= thr;
                    if (invert) {
                        fg = !fg;
                    }
                    const size_t idx = static_cast<size_t>(z) + static_cast<size_t>(y) * s + static_cast<size_t>(x) * s * s;
                    binary[idx] = fg ? 0 : 1; // distance to foreground (zeros)
                }
            }
        }

        float* edt = edt::binary_edt<uint8_t>(
            binary.data(), s, s, s, 1.0f, 1.0f, 1.0f, false, 1);

        const int low = BORDER;

        for (int z = 0; z < CHUNK_SIZE; ++z) {
            for (int y = 0; y < CHUNK_SIZE; ++y) {
                for (int x = 0; x < CHUNK_SIZE; ++x) {
                    const size_t idx = static_cast<size_t>(z + low) + static_cast<size_t>(y + low) * s +
                                       static_cast<size_t>(x + low) * s * s;
                    float d = edt[idx];
                    if (!std::isfinite(d)) {
                        d = 255.0f;
                    }
                    d = std::clamp(d, 0.0f, 255.0f);
                    small(z, y, x) = static_cast<uint8_t>(std::lround(d));
                }
            }
        }

        delete[] edt;
    }

    float threshold = 170.0f;
    bool invert = false;
    std::string UNIQUE_ID_STRING;
};

static void flood_fill_unapproved(const cv::Mat_<uchar>& approved,
                                  const cv::Vec2i& seed,
                                  cv::Mat_<uchar>& interior)
{
    if (!point_in_bounds(approved, seed)) {
        return;
    }
    if (approved(seed) != 0) {
        return;
    }

    std::queue<cv::Vec2i> queue;
    queue.push(seed);
    interior(seed) = 1;

    const int dr[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const int dc[] = {-1, 0, 1, -1, 1, -1, 0, 1};

    while (!queue.empty()) {
        const cv::Vec2i cur = queue.front();
        queue.pop();
        for (int i = 0; i < 8; ++i) {
            const cv::Vec2i next{cur[0] + dr[i], cur[1] + dc[i]};
            if (!point_in_bounds(approved, next)) {
                continue;
            }
            if (approved(next) != 0 || interior(next) != 0) {
                continue;
            }
            interior(next) = 1;
            queue.push(next);
        }
    }
}

static void compute_boundary_from_interior(const cv::Mat_<uchar>& interior,
                                           cv::Mat_<uchar>& boundary)
{
    boundary = cv::Mat_<uchar>(interior.size(), static_cast<uchar>(0));

    const int dr[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const int dc[] = {-1, 0, 1, -1, 1, -1, 0, 1};

    for (int r = 0; r < interior.rows; ++r) {
        for (int c = 0; c < interior.cols; ++c) {
            if (!interior(r, c)) {
                continue;
            }
            bool is_boundary = false;
            for (int i = 0; i < 8; ++i) {
                int nr = r + dr[i];
                int nc = c + dc[i];
                if (nr < 0 || nr >= interior.rows || nc < 0 || nc >= interior.cols ||
                    interior(nr, nc) == 0) {
                    is_boundary = true;
                    break;
                }
            }
            if (is_boundary) {
                boundary(r, c) = 1;
            }
        }
    }
}

// Normal3D placeholder / validity check.
// The fitted normal direction-field stores placeholder/unset normals as the neutral uint8 triplet (128,128,128).
// We treat a trilinear sample as invalid if *any* of the 8 lattice corners is a placeholder.
static inline bool normal3d_trilinear_sample_valid(Chunked3dVec3fFromUint8& dirs, const cv::Vec3d& xyz)
{
    const double zf = xyz[2] * static_cast<double>(dirs._scale);
    const double yf = xyz[1] * static_cast<double>(dirs._scale);
    const double xf = xyz[0] * static_cast<double>(dirs._scale);

    int z0 = static_cast<int>(std::floor(zf));
    int y0 = static_cast<int>(std::floor(yf));
    int x0 = static_cast<int>(std::floor(xf));

    const auto shape = dirs._x.shape(); // z,y,x
    if (!shape.empty()) {
        z0 = std::clamp(z0, 0, std::max(0, shape[0] - 2));
        y0 = std::clamp(y0, 0, std::max(0, shape[1] - 2));
        x0 = std::clamp(x0, 0, std::max(0, shape[2] - 2));
    } else {
        z0 = std::max(0, z0);
        y0 = std::max(0, y0);
        x0 = std::max(0, x0);
    }

    const int z1 = z0 + 1;
    const int y1 = y0 + 1;
    const int x1 = x0 + 1;

    auto is_fill_triplet = [&](int z, int y, int x) -> bool {
        const auto rx = dirs._x.safe_at(z, y, x);
        const auto ry = dirs._y.safe_at(z, y, x);
        const auto rz = dirs._z.safe_at(z, y, x);
        return (rx == 128) && (ry == 128) && (rz == 128);
    };

    const bool any_fill =
        is_fill_triplet(z0, y0, x0) || is_fill_triplet(z1, y0, x0) || is_fill_triplet(z0, y1, x0) || is_fill_triplet(z1, y1, x0) ||
        is_fill_triplet(z0, y0, x1) || is_fill_triplet(z1, y0, x1) || is_fill_triplet(z0, y1, x1) || is_fill_triplet(z1, y1, x1);

    return !any_fill;
}

class PointCorrection {
public:
    struct CorrectionCollection {
        std::vector<cv::Vec3f> tgts_;
        std::vector<cv::Vec2f> grid_locs_;
        std::optional<cv::Vec2f> anchor2d_;  // If set, use this as the 2D grid anchor instead of searching from first point
    };

    PointCorrection() = default;
    PointCorrection(const VCCollection& corrections) {
        const auto& collections = corrections.getAllCollections();
        if (collections.empty()) return;

        for (const auto& pair : collections) {
            const auto& collection = pair.second;
            // Allow collections with anchor2d set even if they have no points (drag-and-drop case)
            if (collection.points.empty() && !collection.anchor2d.has_value()) continue;

            CorrectionCollection new_collection;
            new_collection.anchor2d_ = collection.anchor2d;

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
            if (collection.anchor2d_.has_value()) {
                // Use the provided 2D anchor directly - this is the drag-and-drop case
                cv::Vec2f anchor = collection.anchor2d_.value();

                // Bounds check: skip collections where anchor2d is outside surface bounds
                if (anchor[0] < 0 || anchor[0] >= points.cols ||
                    anchor[1] < 0 || anchor[1] >= points.rows) {
                    std::cout << "Warning: skipping correction with out-of-bounds anchor2d: "
                              << anchor << " (surface size: " << points.cols << "x" << points.rows << ")" << std::endl;
                    continue;
                }

                std::cout << "using provided anchor2d: " << anchor << std::endl;

                // Convert 2D grid location to pointer coordinates
                // pointer coords are relative to center: ptr = grid_loc - center
                // With scale {1,1}, center is {cols/2, rows/2, 0}
                cv::Vec3f ptr = {
                    anchor[0] - points.cols / 2.0f,
                    anchor[1] - points.rows / 2.0f,
                    0.0f
                };

                // Search for all correction points from the anchor position
                for (size_t i = 0; i < collection.tgts_.size(); ++i) {
                    float d = tmp.pointTo(ptr, collection.tgts_[i], 100.0f, 0);
                    cv::Vec3f loc_3d = tmp.loc_raw(ptr);
                    std::cout << "point diff: " << d << loc_3d << std::endl;
                    cv::Vec2f loc = {loc_3d[0], loc_3d[1]};
                    collection.grid_locs_.push_back(loc);
                }
            } else {
                // Original behavior: use first point as anchor
                if (collection.tgts_.empty()) continue;

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

static std::optional<cv::Vec2i> pick_seed_for_collection(
    const PointCorrection::CorrectionCollection& collection,
    const cv::Rect& resume_area,
    const cv::Size& trace_size,
    int resume_pad_x,
    int resume_pad_y)
{
    auto in_trace_bounds = [&](const cv::Vec2i& p) {
        return p[0] >= 0 && p[0] < trace_size.height &&
               p[1] >= 0 && p[1] < trace_size.width;
    };

    if (collection.anchor2d_.has_value()) {
        const cv::Vec2f anchor = collection.anchor2d_.value();
        cv::Vec2i seed{static_cast<int>(std::round(anchor[1])),
                       static_cast<int>(std::round(anchor[0]))};
        if (in_trace_bounds(seed) && resume_area.contains(cv::Point(seed[1], seed[0]))) {
            return seed;
        }
        cv::Vec2i offset_seed{seed[0] + resume_pad_y, seed[1] + resume_pad_x};
        if (in_trace_bounds(offset_seed) && resume_area.contains(cv::Point(offset_seed[1], offset_seed[0]))) {
            return offset_seed;
        }
    }

    if (!collection.grid_locs_.empty()) {
        cv::Vec2f avg(0.0f, 0.0f);
        for (const auto& loc : collection.grid_locs_) {
            avg += loc;
        }
        avg *= (1.0f / static_cast<float>(collection.grid_locs_.size()));
        cv::Vec2i seed{static_cast<int>(std::round(avg[1])),
                       static_cast<int>(std::round(avg[0]))};
        if (in_trace_bounds(seed)) {
            return seed;
        }
    }

    return std::nullopt;
}

struct TraceData {
    TraceData(const std::vector<DirectionField> &direction_fields) : direction_fields(direction_fields) {};
    PointCorrection point_correction;
    const vc::core::util::NormalGridVolume *ngv = nullptr;
    const std::vector<DirectionField> &direction_fields;
    bool cell_reopt_mode = false;
    cv::Mat_<uchar> boundary_mask;
    cv::Mat_<uchar> interior_mask;
    const cv::Mat_<cv::Vec3d>* reopt_anchors = nullptr;
    const cv::Mat_<cv::Vec3d>* reopt_normals = nullptr;
    double reopt_tangent_weight = 10.0;
    double reopt_boundary_weight = 10.0;
    double reopt_boundary_max = 3.0;
    std::shared_ptr<SDTContext> sdt_context;

    // Optional fitted-3D normals direction-field (zarr root with x/<scale>,y/<scale>,z/<scale> datasets)
    std::unique_ptr<Chunked3dVec3fFromUint8> normal3d_field;
    std::unique_ptr<NormalFitQualityWeightField> normal3d_fit_quality;

    Chunked3d<uint8_t, passTroughComputor>* raw_volume = nullptr;
    std::unique_ptr<lineLossDistance> space_line_compute;
    std::unique_ptr<Chunked3d<uint8_t, lineLossDistance>> space_line_volume;
};

class ReferenceClearanceCost {
public:
    ReferenceClearanceCost(const cv::Vec3d& target,
                           double min_clearance,
                           double weight)
        : target_(target),
          min_clearance_(min_clearance),
          weight_(weight) {}

    bool operator()(const double* candidate, double* residual) const {
        if (weight_ <= 0.0 || min_clearance_ <= 0.0) {
            residual[0] = 0.0;
            return true;
        }

        const cv::Vec3d diff{candidate[0] - target_[0],
                             candidate[1] - target_[1],
                             candidate[2] - target_[2]};
        const double dist = cv::norm(diff);
        if (dist >= min_clearance_) {
            residual[0] = 0.0;
        } else {
            residual[0] = weight_ * (min_clearance_ - dist);
        }
        return true;
    }

private:
    cv::Vec3d target_;
    double min_clearance_;
    double weight_;
};

class ReferenceRayOcclusionCost {
public:
    ReferenceRayOcclusionCost(Chunked3d<uint8_t, passTroughComputor>* volume,
                              const cv::Vec3d& target,
                              double threshold,
                              double weight,
                              double step,
                              double max_distance) :
        volume_(volume),
        target_(target),
        threshold_(threshold),
        weight_(weight),
        step_(step > 0.0 ? step : 1.0),
        max_distance_(max_distance)
    {}

    bool operator()(const double* candidate, double* residual) const {
        if (!volume_ || weight_ <= 0.0) {
            residual[0] = 0.0;
            return true;
        }

        const cv::Vec3d start{candidate[0], candidate[1], candidate[2]};
        const double distance = cv::norm(target_ - start);
        if (distance <= 1e-6) {
            residual[0] = 0.0;
            return true;
        }

        if (max_distance_ > 0.0 && distance > max_distance_) {
            residual[0] = 0.0;
            return true;
        }

        const int steps = std::max(1, static_cast<int>(std::ceil(distance / step_)));
        const cv::Vec3d delta = (target_ - start) / static_cast<double>(steps + 1);

        double max_value = std::numeric_limits<double>::lowest();
        bool hit_threshold = false;
        cv::Vec3d current = start;

        const double start_value = sample(start);
        int begin_step = 1;
        if (std::isfinite(start_value) && start_value >= threshold_) {
            bool exited_material = false;
            for (; begin_step <= steps; ++begin_step) {
                current += delta;
                const double value = sample(current);
                if (!std::isfinite(value)) {
                    continue;
                }
                if (value < threshold_) {
                    exited_material = true;
                    ++begin_step;  // start checking one step beyond the exit.
                    break;
                }
            }
            if (!exited_material) {
                residual[0] = 0.0;
                return true;
            }
        } else {
            current = start;
        }

        for (int i = begin_step; i <= steps; ++i) {
            current += delta;
            const double value = sample(current);
            if (!std::isfinite(value)) {
                continue;
            }

            max_value = std::max(max_value, value);
            if (value >= threshold_) {
                hit_threshold = true;
                break;
            }
        }

        if (!hit_threshold) {
            residual[0] = 0.0;
            return true;
        }

        if (!std::isfinite(max_value) || max_value < 0.0) {
            max_value = 0.0;
        }

        const double diff = std::max(0.0, threshold_ - max_value);
        residual[0] = weight_ * diff;
        return true;
    }

private:
    double sample(const cv::Vec3d& xyz) const {
        if (!interp_) {
            interp_ = std::make_unique<CachedChunked3dInterpolator<uint8_t, passTroughComputor>>(*volume_);
        }
        double value = 0.0;
        interp_->Evaluate(xyz[2], xyz[1], xyz[0], &value);
        return value;
    }

    Chunked3d<uint8_t, passTroughComputor>* volume_;
    cv::Vec3d target_;
    double threshold_;
    double weight_;
    double step_;
    double max_distance_;
    mutable std::unique_ptr<CachedChunked3dInterpolator<uint8_t, passTroughComputor>> interp_;
};

struct SDTChunk {
    cv::Vec3i origin;
    cv::Vec3i size;
    std::unique_ptr<float[]> data;
};

struct Vec3iEqual {
    bool operator()(const cv::Vec3i& a, const cv::Vec3i& b) const {
        return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
    }
};

struct SDTContext {
    z5::Dataset* dataset = nullptr;
    ChunkCache<uint8_t>* cache = nullptr;
    int chunk_size = 64;
    float threshold = 1.0f;
    float max_move = 20.0f;
    std::unordered_map<cv::Vec3i, SDTChunk, vec3i_hash, Vec3iEqual> chunks;
    std::mutex mutex;
};

static cv::Vec3i sdt_chunk_origin(const cv::Vec3f& world_pt, int chunk_size)
{
    return cv::Vec3i(
        static_cast<int>(std::floor(world_pt[0] / chunk_size)) * chunk_size,
        static_cast<int>(std::floor(world_pt[1] / chunk_size)) * chunk_size,
        static_cast<int>(std::floor(world_pt[2] / chunk_size)) * chunk_size);
}

static SDTChunk* get_or_compute_sdt_chunk(SDTContext& ctx, const cv::Vec3f& world_pt)
{
    if (!ctx.dataset || !ctx.cache || ctx.chunk_size <= 0) {
        return nullptr;
    }

    const cv::Vec3i origin = sdt_chunk_origin(world_pt, ctx.chunk_size);
    {
        std::lock_guard<std::mutex> lock(ctx.mutex);
        auto it = ctx.chunks.find(origin);
        if (it != ctx.chunks.end()) {
            return &it->second;
        }
    }

    const int cs = ctx.chunk_size;
    const cv::Vec3i size(cs, cs, cs);
    xt::xtensor<uint8_t, 3, xt::layout_type::column_major> binary_data(
        std::array<size_t, 3>{static_cast<size_t>(cs), static_cast<size_t>(cs), static_cast<size_t>(cs)});
    binary_data.fill(0);

    auto shape = ctx.dataset->shape(); // z,y,x
    cv::Vec3i clamped_origin(
        std::max(0, origin[0]),
        std::max(0, origin[1]),
        std::max(0, origin[2]));
    cv::Vec3i clamped_end(
        std::min(static_cast<int>(shape[2]), origin[0] + cs),
        std::min(static_cast<int>(shape[1]), origin[1] + cs),
        std::min(static_cast<int>(shape[0]), origin[2] + cs));
    cv::Vec3i read_size = clamped_end - clamped_origin;

    if (read_size[0] > 0 && read_size[1] > 0 && read_size[2] > 0) {
        cv::Vec3i clamped_origin_zyx(clamped_origin[2], clamped_origin[1], clamped_origin[0]);
        cv::Vec3i read_size_zyx(read_size[2], read_size[1], read_size[0]);
        xt::xtensor<uint8_t, 3, xt::layout_type::column_major> read_buf(
            std::array<size_t, 3>{static_cast<size_t>(read_size_zyx[0]),
                                  static_cast<size_t>(read_size_zyx[1]),
                                  static_cast<size_t>(read_size_zyx[2])});
        readArea3D(read_buf, clamped_origin_zyx, ctx.dataset, ctx.cache);

        const cv::Vec3i offset = clamped_origin - origin;
        for (int z = 0; z < read_size[2]; ++z) {
            for (int y = 0; y < read_size[1]; ++y) {
                for (int x = 0; x < read_size[0]; ++x) {
                    const uint8_t v = read_buf(z, y, x);
                    binary_data(x + offset[0], y + offset[1], z + offset[2]) =
                        static_cast<uint8_t>(v >= ctx.threshold ? 1 : 0);
                }
            }
        }
    }

    const size_t voxels = static_cast<size_t>(cs) * cs * cs;
    std::vector<uint8_t> inverted(voxels);
    for (size_t i = 0; i < voxels; ++i) {
        inverted[i] = binary_data.data()[i] ? 0 : 1;
    }

    float* edt_outside = edt::binary_edt<uint8_t>(
        inverted.data(), cs, cs, cs, 1.0f, 1.0f, 1.0f, false, 1);
    float* edt_inside = edt::binary_edt<uint8_t>(
        binary_data.data(), cs, cs, cs, 1.0f, 1.0f, 1.0f, false, 1);

    SDTChunk chunk;
    chunk.origin = origin;
    chunk.size = size;
    chunk.data = std::make_unique<float[]>(voxels);
    for (size_t i = 0; i < voxels; ++i) {
        chunk.data[i] = binary_data.data()[i] ? -edt_inside[i] : edt_outside[i];
    }

    delete[] edt_outside;
    delete[] edt_inside;

    std::lock_guard<std::mutex> lock(ctx.mutex);
    auto [it, inserted] = ctx.chunks.emplace(origin, std::move(chunk));
    return &it->second;
}

static float sample_sdt(SDTContext& ctx, const cv::Vec3f& world_pt)
{
    SDTChunk* chunk = get_or_compute_sdt_chunk(ctx, world_pt);
    if (!chunk) {
        return 0.0f;
    }
    cv::Vec3f local = world_pt - cv::Vec3f(chunk->origin[0], chunk->origin[1], chunk->origin[2]);
    int x = std::clamp(static_cast<int>(std::round(local[0])), 0, chunk->size[0] - 1);
    int y = std::clamp(static_cast<int>(std::round(local[1])), 0, chunk->size[1] - 1);
    int z = std::clamp(static_cast<int>(std::round(local[2])), 0, chunk->size[2] - 1);
    return chunk->data[static_cast<size_t>(z) * chunk->size[1] * chunk->size[0] +
                       static_cast<size_t>(y) * chunk->size[0] + static_cast<size_t>(x)];
}

class NormalOnlyPenalty {
public:
    NormalOnlyPenalty(cv::Vec3d anchor, cv::Vec3d normal, double weight)
        : anchor_(anchor), normal_(normal), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const candidate, T* residual) const {
        const T dx = candidate[0] - T(anchor_[0]);
        const T dy = candidate[1] - T(anchor_[1]);
        const T dz = candidate[2] - T(anchor_[2]);
        const T dot = dx * T(normal_[0]) + dy * T(normal_[1]) + dz * T(normal_[2]);
        const T tx = dx - dot * T(normal_[0]);
        const T ty = dy - dot * T(normal_[1]);
        const T tz = dz - dot * T(normal_[2]);
        residual[0] = T(weight_) * tx;
        residual[1] = T(weight_) * ty;
        residual[2] = T(weight_) * tz;
        return true;
    }

    static ceres::CostFunction* Create(cv::Vec3d anchor, cv::Vec3d normal, double weight)
    {
        return new ceres::AutoDiffCostFunction<NormalOnlyPenalty, 3, 3>(
            new NormalOnlyPenalty(anchor, normal, weight));
    }

private:
    cv::Vec3d anchor_;
    cv::Vec3d normal_;
    double weight_;
};

class NormalDisplacementClamp {
public:
    NormalDisplacementClamp(cv::Vec3d anchor, cv::Vec3d normal, double max_dist, double weight)
        : anchor_(anchor), normal_(normal), max_dist_(max_dist), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const candidate, T* residual) const {
        const T dx = candidate[0] - T(anchor_[0]);
        const T dy = candidate[1] - T(anchor_[1]);
        const T dz = candidate[2] - T(anchor_[2]);
        const T dot = dx * T(normal_[0]) + dy * T(normal_[1]) + dz * T(normal_[2]);
        const T abs_dist = sqrt(dot * dot + T(1e-12));
        const T excess = abs_dist - T(max_dist_);

        T softplus_val;
        if (val(excess) > T(20)) {
            softplus_val = excess;
        } else if (val(excess) < T(-20)) {
            softplus_val = T(0);
        } else {
            softplus_val = log(T(1) + exp(excess));
        }

        residual[0] = T(weight_) * softplus_val;
        return true;
    }

    static ceres::CostFunction* Create(cv::Vec3d anchor, cv::Vec3d normal, double max_dist, double weight)
    {
        return new ceres::AutoDiffCostFunction<NormalDisplacementClamp, 1, 3>(
            new NormalDisplacementClamp(anchor, normal, max_dist, weight));
    }

private:
    cv::Vec3d anchor_;
    cv::Vec3d normal_;
    double max_dist_;
    double weight_;
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
    NORMAL3DLINE,
    SDIR,
    CORRECTION,
    REFERENCE_RAY,
    SURFACE_SDT,
    SPACELINE,
    COUNT
};

struct LossSettings {
    std::vector<float> w = std::vector<float>(LossType::COUNT, 0.0f);
    std::vector<cv::Mat_<float>> w_mats = std::vector<cv::Mat_<float>>(LossType::COUNT);

    LossSettings() {
        w[LossType::SNAP] = 0.1f;
        w[LossType::NORMAL] = 10.0f;
        w[LossType::NORMAL3DLINE] = 0.0f;
        w[LossType::STRAIGHT] = 0.2;
        w[LossType::DIST] = 1.0f;
        w[LossType::DIRECTION] = 1.0f;
        w[LossType::SDIR] = 1.0f;
        w[LossType::CORRECTION] = 1.0f;
        w[LossType::REFERENCE_RAY] = 0.0f;
        w[LossType::SURFACE_SDT] = 0.0f;
        w[LossType::SPACELINE] = 0.0f;
    }

    struct ReferenceRaycastSettings {
        QuadSurface* surface = nullptr;
        double voxel_threshold = 1.0;
        double sample_step = 1.0;
        double max_distance = 250.0;
        double min_clearance = 4.0;
        double clearance_weight = 1.0;
    } reference_raycast;

    bool reference_raycast_enabled() const {
        return reference_raycast.surface && w[LossType::REFERENCE_RAY] > 0.0f;
    }

    void applyJsonWeights(const nlohmann::json& params) {
        const auto set_weight = [&](const char* key, LossType type) {
            const auto it = params.find(key);
            if (it == params.end() || it->is_null()) {
                return;
            }
            if (it->is_number()) {
                w[type] = static_cast<float>(it->get<double>());
            } else {
                std::cerr << key << " must be numeric" << std::endl;
            }
        };

        set_weight("snap_weight", LossType::SNAP);
        set_weight("normal_weight", LossType::NORMAL);
        set_weight("normal3dline_weight", LossType::NORMAL3DLINE);
        set_weight("straight_weight", LossType::STRAIGHT);
        set_weight("dist_weight", LossType::DIST);
        set_weight("direction_weight", LossType::DIRECTION);
        set_weight("sdir_weight", LossType::SDIR);
        set_weight("correction_weight", LossType::CORRECTION);
        set_weight("reference_ray_weight", LossType::REFERENCE_RAY);
        set_weight("sdt_weight", LossType::SURFACE_SDT);
        set_weight("space_line_weight", LossType::SPACELINE);
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
    int y_min = -1;
    int y_max = std::numeric_limits<int>::max();
    int x_min = -1;
    int x_max = std::numeric_limits<int>::max();
    // Anti-flipback constraint settings
    float flipback_threshold = 5.0f;  // Allow up to this much inward movement (voxels) before penalty
    float flipback_weight = 1.0f;     // Weight of the anti-flipback loss (0 = disabled)

    int space_line_steps = 8;
    float space_line_threshold = 170.0f;
    bool space_line_invert = false;
};

static std::vector<cv::Vec2i> parse_growth_directions(const nlohmann::json& params)
{
    static const std::vector<cv::Vec2i> kDefaultDirections = {
        {1, 0},    // down / +row
        {0, 1},    // right / +col
        {-1, 0},   // up / -row
        {0, -1},   // left / -col
        {1, 1},    // down-right
        {1, -1},   // down-left
        {-1, 1},   // up-right
        {-1, -1}   // up-left
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

    std::vector<cv::Vec2i> custom;
    custom.reserve(kDefaultDirections.size());
    bool any_valid = false;

    auto append_unique = [&custom](const cv::Vec2i& dir) {
        for (const auto& existing : custom) {
            if (existing[0] == dir[0] && existing[1] == dir[1]) {
                return;
            }
        }
        custom.push_back(dir);
    };

    for (const auto& entry : directions) {
        if (!entry.is_string()) {
            std::cerr << "Ignoring non-string entry in growth_directions" << std::endl;
            continue;
        }

        const std::string value = entry.get<std::string>();
        std::string normalized;
        normalized.reserve(value.size());
        for (char ch : value) {
            const unsigned char uch = static_cast<unsigned char>(ch);
            char lower = static_cast<char>(std::tolower(uch));
            if (lower == '-' || lower == '_' || lower == ' ') {
                continue;
            }
            normalized.push_back(lower);
        }

        if (normalized.empty()) {
            std::cerr << "Empty growth direction entry ignored" << std::endl;
            continue;
        }

        if (normalized == "all" || normalized == "default") {
            return kDefaultDirections;
        }

        auto mark_valid = [&](const cv::Vec2i& dir) {
            append_unique(dir);
            any_valid = true;
        };

        if (normalized == "down") {
            mark_valid({1, 0});
            continue;
        }
        if (normalized == "right") {
            mark_valid({0, 1});
            continue;
        }
        if (normalized == "up") {
            mark_valid({-1, 0});
            continue;
        }
        if (normalized == "left") {
            mark_valid({0, -1});
            continue;
        }
        if (normalized == "downright") {
            mark_valid({1, 1});
            continue;
        }
        if (normalized == "downleft") {
            mark_valid({1, -1});
            continue;
        }
        if (normalized == "upright") {
            mark_valid({-1, 1});
            continue;
        }
        if (normalized == "upleft") {
            mark_valid({-1, -1});
            continue;
        }

        std::cerr << "Unknown growth direction '" << value << "' ignored" << std::endl;
    }

    if (!any_valid) {
        return kDefaultDirections;
    }

    return custom;
}

} // namespace

struct NeuralTracerPointInfo {
    cv::Vec2i best_dir;
    std::optional<cv::Vec2i> best_ortho_pos;
    std::optional<cv::Vec2i> best_diag_pos;
    int max_score;
};

static NeuralTracerPointInfo compute_neural_tracer_point_info(
    const cv::Vec2i& p,
    TraceParameters& trace_params)
{
    auto is_valid = [&](const cv::Vec2i& pt) {
        return point_in_bounds(trace_params.state, pt) && (trace_params.state(pt) & STATE_LOC_VALID);
    };

    NeuralTracerPointInfo result;
    result.max_score = -1;

    const cv::Vec2i dirs[] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    for (const auto& dir : dirs) {
        if (!is_valid(p - dir) || !is_valid(p - 2 * dir)) {  // check if center and prev_u (where u is dir-direction) are valid
            continue;
        }

        const cv::Vec2i center_pos = p - dir;

        const cv::Vec2i ortho_dirs[] = {{-dir[1], dir[0]}, {dir[1], -dir[0]}};

        for (const auto& ortho_dir : ortho_dirs) {

            auto current_ortho_pos = is_valid(center_pos - ortho_dir) ? std::optional{center_pos - ortho_dir} : std::nullopt;
            cv::Vec2i diag_dir;  // vector from prev_diag to center
            if (current_ortho_pos) {
                // if prev_u and prev_v both given, then prev_diag should be adjacent to the gap p (not to center)
                diag_dir = dir - ortho_dir;
            } else {
                // if only prev_u not prev_v given, then prev_diag should be opposite to the gap p along ortho_dir
                diag_dir = dir + ortho_dir;
            }
            auto current_diag_pos = is_valid(center_pos - diag_dir) ? std::optional{center_pos - diag_dir} : std::nullopt;

            int current_score =
                current_ortho_pos && current_diag_pos ? 3 :
                current_ortho_pos ? 2 :
                current_diag_pos ? 1 :
                0;  // only prev_u & center valid

            if (current_score > result.max_score) {
                result.max_score = current_score;
                result.best_dir = dir;
                result.best_ortho_pos = current_ortho_pos;
                result.best_diag_pos = current_diag_pos;
            }
        }
    }

    return result;
}

static std::vector<cv::Vec2i> call_neural_tracer_for_points(
    const std::vector<cv::Vec2i>& points,
    TraceParameters& trace_params,
    NeuralTracerConnection* neural_tracer)
{
    if (!neural_tracer)
        throw std::logic_error("Neural tracer connection is null");

    std::vector<cv::Vec3f> center_xyzs;
    std::vector<std::optional<cv::Vec3f>> prev_u_xyzs, prev_v_xyzs, prev_diag_xyzs;
    std::vector<cv::Vec2i> points_with_valid_dirs;

    for (auto const &p : points) {

        auto const point_info = compute_neural_tracer_point_info(p, trace_params);

        if (point_info.max_score < 1) {
            // we disallow score = -1 since this implies no neighbors found, and score = 0 since this is likely to create long, poorly-supported tendrils
            std::cout << "warning: max_score = " << point_info.max_score << std::endl;
            continue;
        }

        auto get_point = [&](const cv::Vec2i& pos) {
            assert(point_in_bounds(trace_params.state, pos) && (trace_params.state(pos) & STATE_LOC_VALID));
            const auto& p_double_neighbor = trace_params.dpoints(pos);
            return cv::Vec3f(p_double_neighbor[0], p_double_neighbor[1], p_double_neighbor[2]);
        };

        center_xyzs.push_back(get_point(p - point_info.best_dir));
        prev_u_xyzs.push_back(get_point(p - 2 * point_info.best_dir));
        prev_v_xyzs.push_back(point_info.best_ortho_pos.transform(get_point));
        prev_diag_xyzs.push_back(point_info.best_diag_pos.transform(get_point));
        points_with_valid_dirs.push_back(p);
    }

    if (points_with_valid_dirs.empty()) {
        return {};
    }

    auto next_uvs = neural_tracer->get_next_points(center_xyzs, prev_u_xyzs, prev_v_xyzs, prev_diag_xyzs);

    std::vector<cv::Vec2i> successful_points;
    for (int point_idx = 0; point_idx < points_with_valid_dirs.size(); point_idx++) {
        auto const p = points_with_valid_dirs[point_idx];
        const auto& candidates = next_uvs[point_idx].next_u_xyzs;
        if (!candidates.empty() && cv::norm(candidates[0]) > 1e-6) {
            trace_params.dpoints(p) = {candidates[0][0], candidates[0][1], candidates[0][2]};
            trace_params.state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
            successful_points.push_back(p);
        } else {
            std::cout << "warning: no valid next point found at " << p << std::endl;
        }
    }

    return successful_points;
}

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
static int gen_3d_normal_line_loss(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &off, TraceParameters &params, const TraceData &trace_data, const LossSettings &settings);
static int conditional_3d_normal_line_loss(int bit, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint16_t> &loss_status, ceres::Problem &problem, TraceParameters &params, const TraceData &trace_data, const LossSettings &settings);
static int gen_dist_loss(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &off, TraceParameters &params, const LossSettings &settings);
static int gen_space_line_loss(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &off, TraceParameters &params, const TraceData &trace_data, const LossSettings &settings);
static int conditional_space_line_loss(int bit, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint16_t> &loss_status, ceres::Problem &problem, TraceParameters &params, const TraceData &trace_data, const LossSettings &settings);
static int gen_sdirichlet_loss(ceres::Problem &problem, const cv::Vec2i &p,
                               TraceParameters &params, const LossSettings &settings,
                               double sdir_eps_abs, double sdir_eps_rel);
static int conditional_sdirichlet_loss(int bit, const cv::Vec2i &p, cv::Mat_<uint16_t> &loss_status,
                                        ceres::Problem &problem, TraceParameters &params,
                                        const LossSettings &settings, double sdir_eps_abs, double sdir_eps_rel);
static int gen_reference_ray_loss(ceres::Problem &problem, const cv::Vec2i &p,
                                  TraceParameters &params, const TraceData &trace_data, const LossSettings &settings);
static int conditional_reference_ray_loss(int bit, const cv::Vec2i &p, cv::Mat_<uint16_t> &loss_status,
                                          ceres::Problem &problem, TraceParameters &params,
                                          const TraceData &trace_data, const LossSettings &settings);
static int gen_surface_sdt_loss(ceres::Problem &problem, const cv::Vec2i &p,
                                TraceParameters &params, const TraceData &trace_data, const LossSettings &settings);
static int conditional_surface_sdt_loss(int bit, const cv::Vec2i &p, cv::Mat_<uint16_t> &loss_status,
                                        ceres::Problem &problem, TraceParameters &params,
                                        const TraceData &trace_data, const LossSettings &settings);

// Used by conditional losses.
static bool loss_mask(int bit, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint16_t> &loss_status);
static int set_loss_mask(int bit, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint16_t> &loss_status, int set);
static bool loc_valid(int state)
{
    return state & STATE_LOC_VALID;
}

static bool coord_valid(int state)
{
    return (state & STATE_COORD_VALID) || (state & STATE_LOC_VALID);
}

static bool dpoint_valid(const cv::Vec3d& p)
{
    return p[0] != -1.0;
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

    const cv::Vec3d& a = params.dpoints(p+o1);
    const cv::Vec3d& b = params.dpoints(p+o2);
    const cv::Vec3d& c = params.dpoints(p+o3);
    if (!dpoint_valid(a) || !dpoint_valid(b) || !dpoint_valid(c)) {
        return 0;
    }
    if (cv::norm(b - a) < 1e-6 || cv::norm(c - b) < 1e-6) {
        return 0;
    }

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

    if (params.dpoints(p)[0] == -1 || params.dpoints(p+off)[0] == -1)
        return 0;

    problem.AddResidualBlock(DistLoss::Create(params.unit*cv::norm(off),settings(LossType::DIST, p)), nullptr, &params.dpoints(p)[0], &params.dpoints(p+off)[0]);

    return 1;
}

static int gen_space_line_loss(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &off,
    TraceParameters &params, const TraceData &trace_data, const LossSettings &settings)
{
    if (!trace_data.space_line_volume)
        return 0;
    if (!coord_valid(params.state(p)))
        return 0;
    if (!coord_valid(params.state(p+off)))
        return 0;
    if (!dpoint_valid(params.dpoints(p)) || !dpoint_valid(params.dpoints(p+off)))
        return 0;

    const float w = settings(LossType::SPACELINE, p);
    if (w <= 0.0f)
        return 0;
    if (settings.space_line_steps < 2)
        return 0;

    problem.AddResidualBlock(
        SpaceLineLossAcc<uint8_t, lineLossDistance>::Create(*trace_data.space_line_volume,
                                                            settings.space_line_steps,
                                                            w),
        nullptr,
        &params.dpoints(p)[0],
        &params.dpoints(p+off)[0]);

    return 1;
}

static int gen_reference_ray_loss(ceres::Problem &problem, const cv::Vec2i &p,
                                  TraceParameters &params, const TraceData &trace_data, const LossSettings &settings)
{
    if (!settings.reference_raycast_enabled())
        return 0;
    if (!trace_data.raw_volume)
        return 0;
    if (!(params.state(p) & STATE_LOC_VALID))
        return 0;

    const float w = settings(LossType::REFERENCE_RAY, p);
    if (w <= 0.0f)
        return 0;

    const cv::Vec3d& candidate = params.dpoints(p);
    if (candidate[0] == -1.0 && candidate[1] == -1.0 && candidate[2] == -1.0)
        return 0;

    cv::Vec3f ptr = settings.reference_raycast.surface->pointer();
    const cv::Vec3f candidate_f(static_cast<float>(candidate[0]),
                                static_cast<float>(candidate[1]),
                                static_cast<float>(candidate[2]));

    float dist = settings.reference_raycast.surface->pointTo(
        ptr,
        candidate_f,
        std::numeric_limits<float>::max(),
        1000);
    if (dist < 0.0f)
        return 0;
    if (settings.reference_raycast.max_distance > 0.0 && dist > settings.reference_raycast.max_distance)
        return 0;

    const cv::Vec3f nearest = settings.reference_raycast.surface->coord(ptr);
    const cv::Vec3d target{nearest[0], nearest[1], nearest[2]};

    {
        auto* functor = new ReferenceRayOcclusionCost(trace_data.raw_volume,
                                                      target,
                                                      settings.reference_raycast.voxel_threshold,
                                                      static_cast<double>(w),
                                                      settings.reference_raycast.sample_step,
                                                      settings.reference_raycast.max_distance);

        auto* cost = new ceres::NumericDiffCostFunction<ReferenceRayOcclusionCost, ceres::CENTRAL, 1, 3>(functor);
        problem.AddResidualBlock(cost, nullptr, &params.dpoints(p)[0]);
    }

    if (settings.reference_raycast.clearance_weight > 0.0 &&
        settings.reference_raycast.min_clearance > 0.0) {
        auto* clearance_functor = new ReferenceClearanceCost(target,
                                                             settings.reference_raycast.min_clearance,
                                                             settings.reference_raycast.clearance_weight * static_cast<double>(w));
        auto* clearance_cost = new ceres::NumericDiffCostFunction<ReferenceClearanceCost, ceres::CENTRAL, 1, 3>(clearance_functor);
        problem.AddResidualBlock(clearance_cost, nullptr, &params.dpoints(p)[0]);
    }

    return 1;
}

static int conditional_reference_ray_loss(int bit,
                                          const cv::Vec2i &p,
                                          cv::Mat_<uint16_t> &loss_status,
                                          ceres::Problem &problem,
                                          TraceParameters &params,
                                          const TraceData &trace_data,
                                          const LossSettings &settings)
{
    int set = 0;
    if (!loss_mask(bit, p, {0, 0}, loss_status)) {
        set = set_loss_mask(bit, p, {0, 0}, loss_status, gen_reference_ray_loss(problem, p, params, trace_data, settings));
    }
    return set;
}

static int gen_surface_sdt_loss(ceres::Problem &problem, const cv::Vec2i &p,
                                TraceParameters &params, const TraceData &trace_data, const LossSettings &settings)
{
    if (!trace_data.cell_reopt_mode || !trace_data.sdt_context) {
        return 0;
    }
    if (trace_data.interior_mask.empty() || !point_in_bounds(trace_data.interior_mask, p) ||
        trace_data.interior_mask(p) == 0) {
        return 0;
    }
    if (!(params.state(p) & STATE_LOC_VALID)) {
        return 0;
    }
    if (!dpoint_valid(params.dpoints(p))) {
        return 0;
    }
    const float w = settings(LossType::SURFACE_SDT, p);
    if (w <= 0.0f) {
        return 0;
    }

    auto sampler = [ctx = trace_data.sdt_context](const cv::Vec3f& sample_pt) -> float {
        if (!ctx) {
            return 0.0f;
        }
        return sample_sdt(*ctx, sample_pt);
    };
    auto* functor = new SignedDistanceToSurfaceCost(sampler, static_cast<double>(w), trace_data.sdt_context->max_move);
    auto* cost = new ceres::NumericDiffCostFunction<SignedDistanceToSurfaceCost, ceres::CENTRAL, 1, 3>(functor);
    problem.AddResidualBlock(cost, nullptr, &params.dpoints(p)[0]);
    return 1;
}

static int conditional_surface_sdt_loss(int bit,
                                        const cv::Vec2i &p,
                                        cv::Mat_<uint16_t> &loss_status,
                                        ceres::Problem &problem,
                                        TraceParameters &params,
                                        const TraceData &trace_data,
                                        const LossSettings &settings)
{
    int set = 0;
    if (!loss_mask(bit, p, {0, 0}, loss_status)) {
        set = set_loss_mask(bit, p, {0, 0}, loss_status, gen_surface_sdt_loss(problem, p, params, trace_data, settings));
    }
    return set;
}

// -------------------------
// helpers used by conditionals (must be before they’re used)
// -------------------------
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

// Order two points along the axis implied by their grid relation.
// - For horizontal edges (same row), order by column.
// - For vertical edges (same col), order by row.
// Falls back to lexicographic (row,col) if neither axis matches.
static inline std::pair<cv::Vec2i, cv::Vec2i> order_p(const cv::Vec2i& p, const cv::Vec2i& q)
{
    if (p[0] == q[0]) { // same row => horizontal edge
        return (p[1] <= q[1]) ? std::make_pair(p, q) : std::make_pair(q, p);
    }
    if (p[1] == q[1]) { // same col => vertical edge
        return (p[0] <= q[0]) ? std::make_pair(p, q) : std::make_pair(q, p);
    }
    // fallback
    return (p[0] < q[0] || (p[0] == q[0] && p[1] <= q[1])) ? std::make_pair(p, q) : std::make_pair(q, p);
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

// -------------------------
// Symmetric Dirichlet losses (definitions)
// -------------------------
static int gen_sdirichlet_loss(ceres::Problem &problem,
                               const cv::Vec2i &p,
                               TraceParameters &params,
                               const LossSettings &settings,
                               double sdir_eps_abs,
                               double sdir_eps_rel)
{
    // Need p, p+u, p+v inside the image; treat (p) as the lower-left of a cell
    const int rows = params.state.rows;
    const int cols = params.state.cols;
    if (p[0] < 0 || p[1] < 0 || p[0] >= rows - 1 || p[1] >= cols - 1) {
        return 0;
    }

    const cv::Vec2i pu = p + cv::Vec2i(0, 1);  // (i, j+1)  u-direction
    const cv::Vec2i pv = p + cv::Vec2i(1, 0);  // (i+1, j)  v-direction

    // All three parameter blocks must be present/valid
    if (!coord_valid(params.state(p)) ||
        !coord_valid(params.state(pu)) ||
        !coord_valid(params.state(pv))) {
        return 0;
    }

    if (!dpoint_valid(params.dpoints(p)) ||
        !dpoint_valid(params.dpoints(pu)) ||
        !dpoint_valid(params.dpoints(pv))) {
        return 0;
    }
    if (cv::norm(params.dpoints(pu) - params.dpoints(p)) < 1e-6 ||
        cv::norm(params.dpoints(pv) - params.dpoints(p)) < 1e-6) {
        return 0;
    }

    const float w = settings(LossType::SDIR, p);
    if (w <= 0.0f) {
        return 0;
    }

    ceres::LossFunction* robust = nullptr;
    robust = new ceres::CauchyLoss(1.0); // cauchy scale = 1.0

    problem.AddResidualBlock(
        SymmetricDirichletLoss::Create(/*unit*/ params.unit,
                                       /*w       */ w,
                                       /*eps_abs */ sdir_eps_abs,
                                       /*eps_rel */ sdir_eps_rel),
        /*loss*/ robust, // Cauchy Loss
        &params.dpoints(p)[0],
        &params.dpoints(pu)[0],
        &params.dpoints(pv)[0]);

    return 1;
}

static int conditional_sdirichlet_loss(int bit,
                                       const cv::Vec2i &p,
                                       cv::Mat_<uint16_t> &loss_status,
                                       ceres::Problem &problem,
                                       TraceParameters &params,
                                       const LossSettings &settings,
                                       double sdir_eps_abs,
                                       double sdir_eps_rel)
{
    int set = 0;
    // One SD term per cell (keyed at p itself)
    if (!loss_mask(bit, p, {0, 0}, loss_status)) {
        set = set_loss_mask(bit, p, {0, 0}, loss_status,
                            gen_sdirichlet_loss(problem, p, params, settings, sdir_eps_abs, sdir_eps_rel));
    }
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

static int conditional_space_line_loss(int bit,
    const cv::Vec2i &p,
    const cv::Vec2i &off,
    cv::Mat_<uint16_t> &loss_status,
    ceres::Problem &problem,
    TraceParameters &params,
    const TraceData &trace_data,
    const LossSettings &settings)
{
    int set = 0;
    if (!loss_mask(bit, p, off, loss_status)) {
        set = set_loss_mask(bit, p, off, loss_status, gen_space_line_loss(problem, p, off, params, trace_data, settings));
    }
    return set;
}

static int gen_3d_normal_line_loss(ceres::Problem &problem,
                                  const cv::Vec2i &p,
                                  const cv::Vec2i &off,
                                  TraceParameters &params,
                                  const TraceData &trace_data,
                                  const LossSettings &settings)
{
    if (!trace_data.normal3d_field) return 0;

    const float w = settings(LossType::NORMAL3DLINE, p);
    if (w <= 0.0f) return 0;

    // We enforce the 90° constraint on the directed segment p_base -> p_off.
    // For consistent disambiguation of the directed normal field, provide a third point:
    // the next quad corner in clockwise direction when walking from base towards off.
    // This third point is treated as non-differentiable inside the loss.
    const cv::Vec2i base = p;
    const cv::Vec2i off_p = p + off;

    // Determine clockwise neighbor in (row,col) grid coordinates.
    // Mapping assumes u = +col (right), v = +row (down).
    // Clockwise around a cell: right -> down -> left -> up.
    cv::Vec2i cw_off;
    if (off[0] == 0 && off[1] == 1) {          // right
        cw_off = cv::Vec2i(1, 0);              // down
    } else if (off[0] == 1 && off[1] == 0) {   // down
        cw_off = cv::Vec2i(0, -1);             // left
    } else if (off[0] == 0 && off[1] == -1) {  // left
        cw_off = cv::Vec2i(-1, 0);             // up
    } else if (off[0] == -1 && off[1] == 0) {  // up
        cw_off = cv::Vec2i(0, 1);              // right
    } else {
        // Only defined for direct 4-neighborhood edges.
        return 0;
    }
    const cv::Vec2i cw_p = base + cw_off;

    if (!coord_valid(params.state(base)) || !coord_valid(params.state(off_p)) || !coord_valid(params.state(cw_p))) {
        return 0;
    }
    if (!dpoint_valid(params.dpoints(base)) ||
        !dpoint_valid(params.dpoints(off_p)) ||
        !dpoint_valid(params.dpoints(cw_p))) {
        return 0;
    }

    problem.AddResidualBlock(
        Normal3DLineLoss::Create(*trace_data.normal3d_field, trace_data.normal3d_fit_quality.get(), w),
        nullptr,
        &params.dpoints(base)[0],
        &params.dpoints(off_p)[0],
        &params.dpoints(cw_p)[0]);

    return 1;
}

static int conditional_3d_normal_line_loss(int bit,
                                          const cv::Vec2i &p,
                                          const cv::Vec2i &off,
                                          cv::Mat_<uint16_t> &loss_status,
                                          ceres::Problem &problem,
                                          TraceParameters &params,
                                          const TraceData &trace_data,
                                          const LossSettings &settings)
{
    if (!trace_data.normal3d_field) return 0;
    int set = 0;
    if (!loss_mask(bit, p, off, loss_status))
        set = set_loss_mask(bit, p, off, loss_status, gen_3d_normal_line_loss(problem, p, off, params, trace_data, settings));
    return set;
}

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
    if (!dpoint_valid(params.dpoints(p)) ||
        !dpoint_valid(params.dpoints(p_tr)) ||
        !dpoint_valid(params.dpoints(p_bl)) ||
        !dpoint_valid(params.dpoints(p_br))) {
        return 0;
    }

    int count = 0;
    // int i = 1;
    for (int i = 0; i < 3; ++i) { // For each plane
        // bool direction_aware = (i == 0); // XY plane

        bool direction_aware = false; // this is not that simple ...
        // Loss with p as base point A
        problem.AddResidualBlock(NormalConstraintPlane::Create(*trace_data.ngv, i, settings(LossType::NORMAL, p), settings(LossType::SNAP, p), trace_data.normal3d_fit_quality.get(), direction_aware, settings.z_min, settings.z_max), nullptr, pA, pB1, pB2, pC);
        // Loss with p_br as base point A
        problem.AddResidualBlock(NormalConstraintPlane::Create(*trace_data.ngv, i, settings(LossType::NORMAL, p), settings(LossType::SNAP, p), trace_data.normal3d_fit_quality.get(), direction_aware, settings.z_min, settings.z_max), nullptr, pC, pB2, pB1, pA);
        // Loss with p_tr as base point A
        problem.AddResidualBlock(NormalConstraintPlane::Create(*trace_data.ngv, i, settings(LossType::NORMAL, p), settings(LossType::SNAP, p), trace_data.normal3d_fit_quality.get(), direction_aware, settings.z_min, settings.z_max), nullptr, pB1, pC, pA, pB2);
        // Loss with p_bl as base point A
        problem.AddResidualBlock(NormalConstraintPlane::Create(*trace_data.ngv, i, settings(LossType::NORMAL, p), settings(LossType::SNAP, p), trace_data.normal3d_fit_quality.get(), direction_aware, settings.z_min, settings.z_max), nullptr, pB2, pA, pC, pB1);
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
static int gen_corr_loss(ceres::Problem &problem, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &loc, TraceData& trace_data, const LossSettings &settings);
static int conditional_corr_loss(int bit, const cv::Vec2i &p, cv::Mat_<uint16_t> &loss_status,
                                 ceres::Problem &problem, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &loc, TraceData& trace_data, const LossSettings &settings);

static int add_losses(ceres::Problem &problem, const cv::Vec2i &p, TraceParameters &params,
    const TraceData &trace_data, const LossSettings &settings, int flags = LOSS_STRAIGHT | LOSS_DIST)
{
    //generate losses for point p
    int count = 0;

    if (p[0] < 2 || p[1] < 2 || p[1] >= params.state.cols-2 || p[0] >= params.state.rows-2)
        throw std::runtime_error("point too close to problem border!");

    if (flags & LOSS_STRAIGHT) {
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
    }

    if (flags & LOSS_DIST) {
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
    }

    if (flags & LOSS_SPACELINE) {
        // direct neighbors only (4-neighborhood)
        count += gen_space_line_loss(problem, p, {0,-1}, params, trace_data, settings);
        count += gen_space_line_loss(problem, p, {0,1}, params, trace_data, settings);
        count += gen_space_line_loss(problem, p, {-1,0}, params, trace_data, settings);
        count += gen_space_line_loss(problem, p, {1,0}, params, trace_data, settings);
    }

    if (flags & LOSS_NORMALSNAP) {
        //gridstore normals
        count += gen_normal_loss(problem, p                   , params, trace_data, settings);
        count += gen_normal_loss(problem, p + cv::Vec2i(-1,-1), params, trace_data, settings);
        count += gen_normal_loss(problem, p + cv::Vec2i( 0,-1), params, trace_data, settings);
        count += gen_normal_loss(problem, p + cv::Vec2i(-1, 0), params, trace_data, settings);
    }

    if (flags & LOSS_SDIR) {
        //symmetric dirichlet
        count += gen_sdirichlet_loss(problem, p, params, settings, /*eps_abs=*/1e-8, /*eps_rel=*/1e-2);
        count += gen_sdirichlet_loss(problem, p + cv::Vec2i(-1, 0), params, settings, 1e-8, 1e-2);
        count += gen_sdirichlet_loss(problem, p + cv::Vec2i( 0,-1), params, settings, 1e-8, 1e-2);
    }

    if (flags & LOSS_3DNORMALLINE) {
        // one per edge (use +u and +v edges only)
        count += gen_3d_normal_line_loss(problem, p, cv::Vec2i(0, 1), params, trace_data, settings);
        count += gen_3d_normal_line_loss(problem, p, cv::Vec2i(1, 0), params, trace_data, settings);
    }

    count += gen_reference_ray_loss(problem, p, params, trace_data, settings);
    count += gen_surface_sdt_loss(problem, p, params, trace_data, settings);

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
static int gen_corr_loss(ceres::Problem &problem, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &dpoints, TraceData& trace_data, const LossSettings &settings)
{
    const float weight = settings(LossType::CORRECTION, p);
    if (!trace_data.point_correction.isValid() || weight <= 0.0f) {
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

    auto points_correction_loss = new PointsCorrectionLoss(filtered_tgts, filtered_grid_locs, quad_loc_int, weight);
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
    ceres::Problem &problem, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &out, TraceData& trace_data, const LossSettings &settings)
{
    if (!trace_data.point_correction.isValid()) return 0;
    int set = 0;
    if (!loss_mask(bit, p, {0,0}, loss_status))
        set = set_loss_mask(bit, p, {0,0}, loss_status, gen_corr_loss(problem, p, state, out, trace_data, settings));
    return set;
};

// Compute local surface normal from neighboring 3D points using cross product of tangent vectors.
// Returns normalized normal vector, or zero vector if insufficient neighbors.
// If surface_normals is provided and has valid neighbors, orients the result to be consistent.
static cv::Vec3d compute_surface_normal_at(
    const cv::Vec2i& p,
    const cv::Mat_<cv::Vec3d>& dpoints,
    const cv::Mat_<uchar>& state,
    const cv::Mat_<cv::Vec3d>* surface_normals = nullptr)
{
    auto is_valid = [&](const cv::Vec2i& pt) {
        return pt[0] >= 0 && pt[0] < dpoints.rows &&
               pt[1] >= 0 && pt[1] < dpoints.cols &&
               (state(pt) & STATE_LOC_VALID);
    };

    // Try to get horizontal and vertical tangents
    cv::Vec3d tangent_h(0,0,0), tangent_v(0,0,0);
    bool has_h = false, has_v = false;

    // Horizontal tangent
    cv::Vec2i left = {p[0], p[1] - 1};
    cv::Vec2i right = {p[0], p[1] + 1};
    if (is_valid(left) && is_valid(right)) {
        tangent_h = dpoints(right) - dpoints(left);
        has_h = true;
    }

    // Vertical tangent
    cv::Vec2i up = {p[0] - 1, p[1]};
    cv::Vec2i down = {p[0] + 1, p[1]};
    if (is_valid(up) && is_valid(down)) {
        tangent_v = dpoints(down) - dpoints(up);
        has_v = true;
    }

    if (!has_h || !has_v) {
        return cv::Vec3d(0,0,0);
    }

    cv::Vec3d normal = tangent_h.cross(tangent_v);
    double len = cv::norm(normal);
    if (len < 1e-9) {
        return cv::Vec3d(0,0,0);
    }
    normal /= len;

    // Orient consistently with neighbors if surface_normals provided
    if (surface_normals) {
        static const cv::Vec2i neighbor_offsets[] = {{-1,0}, {1,0}, {0,-1}, {0,1}};
        for (const auto& off : neighbor_offsets) {
            cv::Vec2i neighbor = p + off;
            if (is_valid(neighbor)) {
                cv::Vec3d neighbor_normal = (*surface_normals)(neighbor);
                if (cv::norm(neighbor_normal) > 0.5) {
                    // Flip if pointing opposite to neighbor
                    if (normal.dot(neighbor_normal) < 0) {
                        normal = -normal;
                    }
                    break;  // Use first valid neighbor for orientation
                }
            }
        }
    }

    return normal;
}

// Compute and store oriented surface normal for a point
// Uses existing neighbor normals for consistent orientation
static void update_surface_normal(
    const cv::Vec2i& p,
    const cv::Mat_<cv::Vec3d>& dpoints,
    const cv::Mat_<uchar>& state,
    cv::Mat_<cv::Vec3d>& surface_normals)
{
    cv::Vec3d normal = compute_surface_normal_at(p, dpoints, state, &surface_normals);
    if (cv::norm(normal) > 0.5) {
        surface_normals(p) = normal;
    }
}

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

    // space-line loss (4-neighborhood)
    count += conditional_space_line_loss(7, p, {0,-1}, loss_status, problem, params, trace_data, settings);
    count += conditional_space_line_loss(7, p, {0,1}, loss_status, problem, params, trace_data, settings);
    count += conditional_space_line_loss(8, p, {-1,0}, loss_status, problem, params, trace_data, settings);
    count += conditional_space_line_loss(8, p, {1,0}, loss_status, problem, params, trace_data, settings);

    //symmetric dirichlet
    count += conditional_sdirichlet_loss(6, p,                    loss_status, problem, params, settings, /*eps_abs=*/1e-8, /*eps_rel=*/1e-2);
    count += conditional_sdirichlet_loss(6, p + cv::Vec2i(-1, 0), loss_status, problem, params, settings, 1e-8, 1e-2);
    count += conditional_sdirichlet_loss(6, p + cv::Vec2i( 0,-1), loss_status, problem, params, settings, 1e-8, 1e-2);

    //normal field
    count += conditional_direction_loss(9, p, 1, loss_status, problem, params.state, params.dpoints, settings, trace_data.direction_fields);
    count += conditional_direction_loss(9, p, -1, loss_status, problem, params.state, params.dpoints, settings, trace_data.direction_fields);

    //gridstore normals
    count += conditional_normal_loss(10, p                   , loss_status, problem, params, trace_data, settings);
    count += conditional_normal_loss(10, p + cv::Vec2i(-1,-1), loss_status, problem, params, trace_data, settings);
    count += conditional_normal_loss(10, p + cv::Vec2i( 0,-1), loss_status, problem, params, trace_data, settings);
    count += conditional_normal_loss(10, p + cv::Vec2i(-1, 0), loss_status, problem, params, trace_data, settings);

    // fitted 3D normals: edge-perpendicularity (once per edge)
    // Add for all edges touching p; the mask makes sure each undirected edge is only added once.
    count += conditional_3d_normal_line_loss(13, p, cv::Vec2i(0, 1),  loss_status, problem, params, trace_data, settings);
    count += conditional_3d_normal_line_loss(13, p, cv::Vec2i(0, -1), loss_status, problem, params, trace_data, settings);
    count += conditional_3d_normal_line_loss(14, p, cv::Vec2i(1, 0),  loss_status, problem, params, trace_data, settings);
    count += conditional_3d_normal_line_loss(14, p, cv::Vec2i(-1, 0), loss_status, problem, params, trace_data, settings);

    //snapping
    count += conditional_corr_loss(11, p,                    loss_status, problem, params.state, params.dpoints, trace_data, settings);
    count += conditional_corr_loss(11, p + cv::Vec2i(-1,-1), loss_status, problem, params.state, params.dpoints, trace_data, settings);
    count += conditional_corr_loss(11, p + cv::Vec2i( 0,-1), loss_status, problem, params.state, params.dpoints, trace_data, settings);
    count += conditional_corr_loss(11, p + cv::Vec2i(-1, 0), loss_status, problem, params.state, params.dpoints, trace_data, settings);

    count += conditional_reference_ray_loss(15, p, loss_status, problem, params, trace_data, settings);
    count += conditional_surface_sdt_loss(12, p, loss_status, problem, params, trace_data, settings);

    return count;
}


template <typename T>
void masked_blur(cv::Mat_<T>& img, const cv::Mat_<uchar>& mask) {
    cv::Mat_<T> fw_pass = img.clone();
    cv::Mat_<T> bw_pass = img.clone();

    // Initial forward pass (top-left neighbors)
    for (int y = 1; y < img.rows; ++y) {
        for (int x = 1; x < img.cols - 1; ++x) {
            if (mask(y, x) == 0) {
                T val = (fw_pass(y - 1, x) + fw_pass(y, x - 1) + fw_pass(y - 1, x - 1) + fw_pass(y - 1, x + 1)) * 0.25;
                fw_pass(y, x) = val;
            }
        }
    }

    // Initial backward pass (bottom-right neighbors)
    for (int y = img.rows - 2; y >= 0; --y) {
        for (int x = img.cols - 2; x >= 1; --x) {
            if (mask(y, x) == 0) {
                T val = (bw_pass(y + 1, x) + bw_pass(y, x + 1) + bw_pass(y + 1, x + 1) + bw_pass(y + 1, x - 1)) * 0.25;
                bw_pass(y, x) = val;
            }
        }
    }

    // Average initial passes
    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            if (mask(y, x) == 0) {
                img(y, x) = (fw_pass(y, x) + bw_pass(y, x)) * 0.5;
            }
        }
    }

    // 4 additional passes
    for (int i = 0; i < 4; ++i) {
        // Forward pass
        for (int y = 1; y < img.rows; ++y) {
            for (int x = 1; x < img.cols - 1; ++x) {
                if (mask(y, x) == 0) {
                    T val = (img(y, x) + img(y - 1, x) + img(y, x - 1) + img(y - 1, x - 1) + img(y - 1, x + 1)) * 0.2;
                    img(y, x) = val;
                }
            }
        }
        // Backward pass
        for (int y = img.rows - 2; y >= 0; --y) {
            for (int x = img.cols - 2; x >= 1; --x) {
                if (mask(y, x) == 0) {
                    T val = (img(y, x) + img(y + 1, x) + img(y, x + 1) + img(y + 1, x + 1) + img(y + 1, x - 1)) * 0.2;
                    img(y, x) = val;
                }
            }
        }
    }
}

static void local_optimization(const cv::Rect &roi, const cv::Mat_<uchar> &mask, TraceParameters &params, const TraceData &trace_data, LossSettings &settings, int flags);

static bool resample_inside_boundary(TraceParameters& params,
                                     const TraceData& trace_data,
                                     LossSettings& base_settings)
{
    if (!trace_data.cell_reopt_mode || trace_data.interior_mask.empty()) {
        return false;
    }

    std::vector<cv::Point> interior_points;
    cv::findNonZero(trace_data.interior_mask, interior_points);
    if (interior_points.empty()) {
        return false;
    }

    cv::Rect roi = cv::boundingRect(interior_points);
    if (roi.width < 3 || roi.height < 3) {
        return false;
    }

    cv::Mat_<uchar> local_mask(roi.size(), static_cast<uchar>(1));
    for (int y = 0; y < roi.height; ++y) {
        for (int x = 0; x < roi.width; ++x) {
            const cv::Vec2i grid{roi.y + y, roi.x + x};
            if (!(params.state(grid) & STATE_LOC_VALID)) {
                local_mask(y, x) = 1;
                continue;
            }
            if (!point_in_bounds(trace_data.interior_mask, grid) || trace_data.interior_mask(grid) == 0) {
                local_mask(y, x) = 1;
                continue;
            }
            if (!trace_data.boundary_mask.empty() && trace_data.boundary_mask(grid) != 0) {
                local_mask(y, x) = 1;
                continue;
            }
            local_mask(y, x) = 0;
        }
    }

    cv::Mat_<cv::Vec3d> dpoints_roi = params.dpoints(roi);
    masked_blur(dpoints_roi, local_mask);

    LossSettings resample_settings = base_settings;
    resample_settings[SNAP] = 0.0f;
    resample_settings[DIST] *= 0.3f;
    resample_settings[STRAIGHT] *= 0.1f;
    resample_settings[NORMAL] *= 0.1f;
    resample_settings[SURFACE_SDT] = 0.0f;
    int flags = LOSS_DIST | LOSS_STRAIGHT | LOSS_NORMALSNAP;
    if (trace_data.space_line_volume && resample_settings.w[LossType::SPACELINE] > 0.0f) {
        flags |= LOSS_SPACELINE;
    }
    local_optimization(roi, local_mask, params, trace_data, resample_settings, flags);
    return true;
}

static void add_cell_reopt_constraints_roi(ceres::Problem& problem,
                                           const cv::Rect& roi,
                                           TraceParameters& params,
                                           const TraceData& trace_data)
{
    if (!trace_data.cell_reopt_mode || trace_data.interior_mask.empty() ||
        !trace_data.reopt_anchors || !trace_data.reopt_normals) {
        return;
    }

    for (int y = 0; y < roi.height; ++y) {
        for (int x = 0; x < roi.width; ++x) {
            const cv::Vec2i grid{roi.y + y, roi.x + x};
            if (!(params.state(grid) & STATE_LOC_VALID)) {
                continue;
            }
            if (!point_in_bounds(trace_data.interior_mask, grid) || trace_data.interior_mask(grid) == 0) {
                continue;
            }
            const cv::Vec3d& anchor = (*trace_data.reopt_anchors)(grid);
            const cv::Vec3d& normal = (*trace_data.reopt_normals)(grid);
            if (anchor[0] < 0.0 || cv::norm(normal) < 0.5) {
                continue;
            }
            problem.AddResidualBlock(NormalOnlyPenalty::Create(anchor, normal, trace_data.reopt_tangent_weight),
                                     nullptr,
                                     &params.dpoints(grid)[0]);
            if (!trace_data.boundary_mask.empty() && trace_data.boundary_mask(grid) != 0) {
                problem.AddResidualBlock(
                    NormalDisplacementClamp::Create(anchor, normal, trace_data.reopt_boundary_max, trace_data.reopt_boundary_weight),
                    nullptr,
                    &params.dpoints(grid)[0]);
            }
        }
    }
}

static void add_cell_reopt_constraints_radius(ceres::Problem& problem,
                                              int radius,
                                              const cv::Vec2i& center,
                                              TraceParameters& params,
                                              const TraceData& trace_data)
{
    if (!trace_data.cell_reopt_mode || trace_data.interior_mask.empty() ||
        !trace_data.reopt_anchors || !trace_data.reopt_normals) {
        return;
    }

    for (int oy = std::max(center[0] - radius, 0); oy <= std::min(center[0] + radius, params.dpoints.rows - 1); ++oy) {
        for (int ox = std::max(center[1] - radius, 0); ox <= std::min(center[1] + radius, params.dpoints.cols - 1); ++ox) {
            const cv::Vec2i grid{oy, ox};
            if (cv::norm(center - grid) > radius) {
                continue;
            }
            if (!(params.state(grid) & STATE_LOC_VALID)) {
                continue;
            }
            if (!point_in_bounds(trace_data.interior_mask, grid) || trace_data.interior_mask(grid) == 0) {
                continue;
            }
            const cv::Vec3d& anchor = (*trace_data.reopt_anchors)(grid);
            const cv::Vec3d& normal = (*trace_data.reopt_normals)(grid);
            if (anchor[0] < 0.0 || cv::norm(normal) < 0.5) {
                continue;
            }
            problem.AddResidualBlock(NormalOnlyPenalty::Create(anchor, normal, trace_data.reopt_tangent_weight),
                                     nullptr,
                                     &params.dpoints(grid)[0]);
            if (!trace_data.boundary_mask.empty() && trace_data.boundary_mask(grid) != 0) {
                problem.AddResidualBlock(
                    NormalDisplacementClamp::Create(anchor, normal, trace_data.reopt_boundary_max, trace_data.reopt_boundary_weight),
                    nullptr,
                    &params.dpoints(grid)[0]);
            }
        }
    }
}

//optimize within a radius, setting edge points to constant
static bool inpaint(const cv::Rect &roi, const cv::Mat_<uchar> &mask, TraceParameters &params, const TraceData &trace_data)
{
    // check that a two pixel border is 1
    for (int y = 0; y < roi.height; ++y) {
        for (int x = 0; x < roi.width; ++x) {
            if (y < 2 || y >= roi.height - 2 || x < 2 || x >= roi.width - 2) {
                if (mask(y, x) == 0) {
                    return false;
                }
            }
        }
    }

    cv::Mat_<cv::Vec3d> dpoints_roi = params.dpoints(roi);
    masked_blur(dpoints_roi, mask);

    ceres::Problem problem;
    LossSettings settings;
    settings[DIST] = 1.0;
    settings[STRAIGHT] = 1.0;

    for (int y = 0; y < roi.height; ++y) {
        for (int x = 0; x < roi.width; ++x) {
            if (!mask(y, x)) {
                params.state(roi.y + y, roi.x + x) = STATE_LOC_VALID | STATE_COORD_VALID;
            }
        }
    }

    LossSettings base;
    base[NORMAL] = 10.0;
    LossSettings nosnap = base;
    nosnap[SNAP] = 0;
    local_optimization(roi, mask, params, trace_data, nosnap, LOSS_DIST | LOSS_STRAIGHT);
    local_optimization(roi, mask, params, trace_data, nosnap, LOSS_DIST | LOSS_STRAIGHT | LOSS_NORMALSNAP);

    LossSettings lowsnap = base;
    lowsnap[SNAP] = 0.01*base[SNAP];
    local_optimization(roi, mask, params, trace_data, lowsnap, LOSS_DIST | LOSS_STRAIGHT | LOSS_NORMALSNAP);
    lowsnap[SNAP] = 0.1*base[SNAP];
    local_optimization(roi, mask, params, trace_data, lowsnap, LOSS_DIST | LOSS_STRAIGHT | LOSS_NORMALSNAP);
    lowsnap[SNAP] = base[SNAP];
    local_optimization(roi, mask, params, trace_data, lowsnap, LOSS_DIST | LOSS_STRAIGHT | LOSS_NORMALSNAP);
    LossSettings default_settings;
    local_optimization(roi, mask, params, trace_data, default_settings, LOSS_DIST | LOSS_STRAIGHT | LOSS_NORMALSNAP);

    return true;
}


//optimize within a radius, setting edge points to constant
static void local_optimization(const cv::Rect &roi, const cv::Mat_<uchar> &mask, TraceParameters &params, const TraceData &trace_data, LossSettings &settings, int flags)
{
    ceres::Problem problem;
    const bool use_cell_reopt_mask = trace_data.cell_reopt_mode && !trace_data.interior_mask.empty();

    for (int y = 2; y < roi.height - 2; ++y) {
        for (int x = 2; x < roi.width - 2; ++x) {
            // if (!mask(y, x)) {
                add_losses(problem, {roi.y + y, roi.x + x}, params, trace_data, settings, flags);
            // }
        }
    }

    add_cell_reopt_constraints_roi(problem, roi, params, trace_data);

    for (int y = 0; y < roi.height; ++y) {
        for (int x = 0; x < roi.width; ++x) {
            bool fixed = mask(y, x) != 0;
            if (use_cell_reopt_mask) {
                const cv::Vec2i grid{roi.y + y, roi.x + x};
                if (!point_in_bounds(trace_data.interior_mask, grid) || trace_data.interior_mask(grid) == 0) {
                    fixed = true;
                }
            }
            if (fixed && problem.HasParameterBlock(&params.dpoints.at<cv::Vec3d>(roi.y + y, roi.x + x)[0])) {
                problem.SetParameterBlockConstant(&params.dpoints.at<cv::Vec3d>(roi.y + y, roi.x + x)[0]);
            }
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 10000;
    // options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // std::cout << "inpaint solve " << summary.BriefReport() << std::endl;

    // cv::imwrite("opt_mask.tif", mask);
}

struct LocalOptimizationConfig {
    int max_iterations = 1000;
    bool use_dense_qr = false;
};

// Configuration for anti-flipback constraint
// This prevents the surface from flipping back through itself during optimization
struct AntiFlipbackConfig {
    const cv::Mat_<cv::Vec3d>* anchors = nullptr;        // Positions before optimization
    const cv::Mat_<cv::Vec3d>* surface_normals = nullptr; // Consistently oriented surface normals
    double threshold = 5.0;   // Allow up to this much inward movement (voxels) before penalty kicks in
    double weight = 1.0;       // Weight of the anti-flipback loss when activated
};

struct LocalOptTimingAccumulator {
    std::atomic<double> problem_setup{0.0};
    std::atomic<double> anti_flipback{0.0};
    std::atomic<double> cell_reopt{0.0};
    std::atomic<double> param_fixup{0.0};
    std::atomic<double> ceres_solve{0.0};
    std::atomic<int> total_solves{0};

    static void atomic_add(std::atomic<double>& target, double value) {
        double old_val = target.load(std::memory_order_relaxed);
        while (!target.compare_exchange_weak(old_val, old_val + value,
               std::memory_order_relaxed, std::memory_order_relaxed)) {}
    }

    void accumulate(double ps, double af, double cr, double pf, double cs) {
        atomic_add(problem_setup, ps);
        atomic_add(anti_flipback, af);
        atomic_add(cell_reopt, cr);
        atomic_add(param_fixup, pf);
        atomic_add(ceres_solve, cs);
        total_solves.fetch_add(1, std::memory_order_relaxed);
    }

    void print(double wall_time) const {
        int n = total_solves.load();
        double ps = problem_setup.load();
        double af = anti_flipback.load();
        double cr = cell_reopt.load();
        double pf = param_fixup.load();
        double cs = ceres_solve.load();
        double total_cpu = ps + af + cr + pf + cs;
        printf("\nresume opt local timing breakdown:\n");
        printf("  problem setup:          %8.3fs  (%5.1f%%)\n", ps, 100.0*ps/total_cpu);
        printf("  anti-flipback losses:   %8.3fs  (%5.1f%%)\n", af, 100.0*af/total_cpu);
        printf("  cell reopt constraints: %8.3fs  (%5.1f%%)\n", cr, 100.0*cr/total_cpu);
        printf("  param block fixup:      %8.3fs  (%5.1f%%)\n", pf, 100.0*pf/total_cpu);
        printf("  ceres solve:            %8.3fs  (%5.1f%%)\n", cs, 100.0*cs/total_cpu);
        printf("  total cpu time:         %8.3fs\n", total_cpu);
        printf("  total wall time:        %8.3fs\n", wall_time);
        printf("  total solves:           %d\n", n);
        if (n > 0)
            printf("  avg solve time:         %8.6fs\n", total_cpu / n);
    }
};

static float local_optimization(int radius, const cv::Vec2i &p, TraceParameters &params,
    TraceData& trace_data, LossSettings &settings, bool quiet = false, bool parallel = false,
    const LocalOptimizationConfig* solver_config = nullptr,
    const AntiFlipbackConfig* flipback_config = nullptr,
    LocalOptTimingAccumulator* timing = nullptr)
{
    // This Ceres problem is parameterised by locs; residuals are progressively added as the patch grows enforcing that
    // all points in the patch are correct distance in 2D vs 3D space, not too high curvature, near surface prediction, etc.
    auto t0 = timing ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};

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

    auto t1 = timing ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};

    // Add anti-flipback loss if configured
    // This penalizes points that move too far in the inward (negative normal) direction
    if (flipback_config && flipback_config->anchors && flipback_config->surface_normals) {
        for(int oy=std::max(p[0]-radius,0);oy<=std::min(p[0]+radius,params.dpoints.rows-1);oy++)
            for(int ox=std::max(p[1]-radius,0);ox<=std::min(p[1]+radius,params.dpoints.cols-1);ox++) {
                cv::Vec2i op = {oy, ox};
                if (cv::norm(p-op) <= radius && (params.state(op) & STATE_LOC_VALID)) {
                    cv::Vec3d anchor = (*flipback_config->anchors)(op);
                    cv::Vec3d normal = (*flipback_config->surface_normals)(op);
                    // Only add loss if we have valid anchor and normal
                    if (cv::norm(normal) > 0.5 && anchor[0] >= 0) {
                        problem.AddResidualBlock(
                            AntiFlipbackLoss::Create(anchor, normal, flipback_config->threshold, flipback_config->weight),
                            nullptr,
                            &params.dpoints(op)[0]);
                    }
                }
            }
    }

    auto t2 = timing ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};

    add_cell_reopt_constraints_radius(problem, radius, p, params, trace_data);

    auto t3 = timing ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};

    for(int oy=std::max(p[0]-r_outer,0);oy<=std::min(p[0]+r_outer,params.dpoints.rows-1);oy++)
        for(int ox=std::max(p[1]-r_outer,0);ox<=std::min(p[1]+r_outer,params.dpoints.cols-1);ox++) {
            cv::Vec2i op = {oy, ox};
            if (!problem.HasParameterBlock(&params.dpoints(op)[0])) {
                continue;
            }
            bool fixed = cv::norm(p-op) > radius;
            if (trace_data.cell_reopt_mode && !trace_data.interior_mask.empty()) {
                if (!point_in_bounds(trace_data.interior_mask, op) || trace_data.interior_mask(op) == 0) {
                    fixed = true;
                }
            }
            if (fixed) {
                problem.SetParameterBlockConstant(&params.dpoints(op)[0]);
            }
        }

    auto t4 = timing ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};

    ceres::Solver::Options options;
    if (solver_config && solver_config->use_dense_qr) {
        options.linear_solver_type = ceres::DENSE_QR;
    } else {
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    }
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = solver_config ? solver_config->max_iterations : 1000;

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

    if (timing) {
        auto t5 = std::chrono::high_resolution_clock::now();
        auto secs = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
        timing->accumulate(secs(t0, t1), secs(t1, t2), secs(t2, t3), secs(t3, t4), secs(t4, t5));
    }

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


QuadSurface *tracer(z5::Dataset *ds, float scale, ChunkCache<uint8_t> *cache, cv::Vec3f origin, const nlohmann::json &params, const std::string &cache_root, float voxelsize, std::vector<DirectionField> const &direction_fields, QuadSurface* resume_surf, const std::filesystem::path& tgt_path, const nlohmann::json& meta_params, const VCCollection &corrections)
{
    std::unique_ptr<NeuralTracerConnection> neural_tracer;
    int pre_neural_gens = 0, neural_batch_size = 1;
    if (params.contains("neural_socket")) {
        std::string socket_path = params["neural_socket"];
        if (!socket_path.empty()) {
            try {
                neural_tracer = std::make_unique<NeuralTracerConnection>(socket_path);
                std::cout << "Neural tracer connection enabled on " << socket_path << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Failed to connect neural tracer: " << e.what() << std::endl;
                throw;
            }
        }
        pre_neural_gens = params.value("pre_neural_generations", 0);
        neural_batch_size = params.value("neural_batch_size", 1);
        if (!neural_tracer) {
            std::cout << "Neural tracer not active" << std::endl;
        }
    }
    TraceData trace_data(direction_fields);
    LossSettings loss_settings;
    loss_settings.applyJsonWeights(params);
    loss_settings.space_line_steps = std::max(2, params.value("space_line_steps", loss_settings.space_line_steps));
    loss_settings.space_line_threshold = std::clamp(static_cast<float>(params.value("space_line_threshold", loss_settings.space_line_threshold)), 0.0f, 255.0f);
    loss_settings.space_line_invert = params.value("space_line_invert", loss_settings.space_line_invert);
    trace_data.cell_reopt_mode = params.value("cell_reopt_mode", false);
    if (trace_data.cell_reopt_mode) {
        std::cout << "Cell reoptimization mode enabled." << std::endl;
        trace_data.reopt_tangent_weight = std::max(0.0, params.value("cell_reopt_tangent_weight", 10.0));
        trace_data.reopt_boundary_weight = std::max(0.0, params.value("cell_reopt_boundary_weight", 10.0));
        trace_data.reopt_boundary_max = std::max(0.0, params.value("cell_reopt_boundary_max", 3.0));
    }
    if (trace_data.cell_reopt_mode && loss_settings.w[LossType::SURFACE_SDT] > 0.0f) {
        auto sdt_context = std::make_shared<SDTContext>();
        sdt_context->dataset = ds;
        sdt_context->cache = cache;
        sdt_context->chunk_size = std::clamp(params.value("sdt_chunk_size", 64), 32, 256);
        sdt_context->threshold = std::clamp(static_cast<float>(params.value("sdt_threshold", 1.0)), 0.0f, 255.0f);
        sdt_context->max_move = std::max(0.0f, static_cast<float>(params.value("sdt_max_move", 20.0)));
        trace_data.sdt_context = std::move(sdt_context);
        std::cout << "Cell reopt SDT enabled (chunk_size=" << trace_data.sdt_context->chunk_size
                  << " threshold=" << trace_data.sdt_context->threshold
                  << " max_move=" << trace_data.sdt_context->max_move << ")" << std::endl;
    }

    // Optional fitted-3D normals field (direction-field zarr root with x/<scale>,y/<scale>,z/<scale>).
    // IMPORTANT: We auto-derive the correct scale factor from dataset shapes and ignore any JSON scale parameter.
    if (params.contains("normal3d_zarr_path") && params["normal3d_zarr_path"].is_string()) {
        try {
            const std::filesystem::path zarr_root = params["normal3d_zarr_path"].get<std::string>();

            // Expect fixed layout: <root>/{x,y,z}/0
            const int scale_level = 0;

            // Read delimiter from x/0/.zarray.
            // Also assert direction-field fill_value uses the neutral (128,128,128) convention.
            // We treat that triplet as "no normal".
            const auto assert_fill_value_128 = [&](const char* axis) {
                const std::filesystem::path zarray_axis = zarr_root / axis / "0" / ".zarray";
                if (!std::filesystem::exists(zarray_axis)) {
                    throw std::runtime_error(std::string("Missing ") + axis + "/0/.zarray under normal3d_zarr_path: " + zarr_root.string());
                }
                nlohmann::json j = nlohmann::json::parse(std::ifstream(zarray_axis));
                if (!j.contains("fill_value")) {
                    throw std::runtime_error(std::string("Missing fill_value in ") + axis + "/0/.zarray under normal3d_zarr_path: " + zarr_root.string());
                }
                const int fv = j["fill_value"].get<int>();
                if (fv != 128) {
                    std::stringstream msg;
                    msg << "normal3d_zarr_path fill_value=" << fv << " for " << axis << "/0; expected 128";
                    throw std::runtime_error(msg.str());
                }
            };

            assert_fill_value_128("x");
            assert_fill_value_128("y");
            assert_fill_value_128("z");

            const std::filesystem::path zarray_x = zarr_root / "x" / "0" / ".zarray";
            nlohmann::json j = nlohmann::json::parse(std::ifstream(zarray_x));
            std::string delim = j.value<std::string>("dimension_separator", ".");

            // Assert the direction-field was aligned by vc_ngrids --align-normals.
            try {
                z5::filesystem::handle::File rootFile(zarr_root);
                z5::filesystem::handle::Group root(rootFile, "");
                nlohmann::json attrs;
                z5::filesystem::readAttributes(root, attrs);
                const bool aligned = attrs.value("align_normals", false);
                if (!aligned) {
                    throw std::runtime_error("normal3d_zarr_path is not marked aligned (missing attrs.align_normals=true); run vc_ngrids --align-normals");
                }
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("Failed normal3d alignment check: ") + e.what());
            }

            // Derive scale purely from shapes: main volume is full-res, normal zarr is downsampled.
            const auto vol_shape_zyx = ds->shape();
            const int vol_z = static_cast<int>(vol_shape_zyx.at(0));
            const int vol_y = static_cast<int>(vol_shape_zyx.at(1));
            const int vol_x = static_cast<int>(vol_shape_zyx.at(2));

            z5::filesystem::handle::Group dirs_group(zarr_root.string(), z5::FileMode::FileMode::r);
            z5::filesystem::handle::Group x_group(dirs_group, "x");
            z5::filesystem::handle::Dataset x_ds_handle(x_group, "0", delim);
            auto x_ds = z5::filesystem::openDataset(x_ds_handle);
            const auto nshape = x_ds->shape();
            if (nshape.size() != 3) {
                throw std::runtime_error("normal3d x/0 dataset is not 3D");
            }
            const int nz = static_cast<int>(nshape.at(0));
            const int ny = static_cast<int>(nshape.at(1));
            const int nx = static_cast<int>(nshape.at(2));
            if (nz <= 0 || ny <= 0 || nx <= 0) {
                throw std::runtime_error("normal3d x/0 dataset has invalid shape");
            }
            // The normal3d dataset is typically generated on a lattice with
            // n = ceil(vol / step), so vol may NOT be divisible by n.
            // Derive the step (=downsample ratio) by rounding vol/n.
            const double rz_f = static_cast<double>(vol_z) / static_cast<double>(nz);
            const double ry_f = static_cast<double>(vol_y) / static_cast<double>(ny);
            const double rx_f = static_cast<double>(vol_x) / static_cast<double>(nx);

            const int rz = std::max(1, static_cast<int>(std::llround(rz_f)));
            const int ry = std::max(1, static_cast<int>(std::llround(ry_f)));
            const int rx = std::max(1, static_cast<int>(std::llround(rx_f)));

            // Be tolerant of off-by-one shape effects from ceil() at the boundary.
            // Require all axes to agree after rounding.
            if (rz != ry || ry != rx) {
                std::stringstream msg;
                msg << "normal3d downsample ratio differs across axes after rounding: "
                    << "rx=" << rx << " ry=" << ry << " rz=" << rz
                    << " (vol=" << vol_x << "x" << vol_y << "x" << vol_z
                    << ", n=" << nx << "x" << ny << "x" << nz << ")";
                throw std::runtime_error(msg.str());
            }
            const int ratio = rx;

            const float scale_factor = 1.0f / static_cast<float>(ratio);

            std::vector<std::unique_ptr<z5::Dataset>> dss;
            for (auto dim : std::string("xyz")) {
                z5::filesystem::handle::Group dim_group(dirs_group, std::string(&dim, 1));
                z5::filesystem::handle::Dataset ds_handle(dim_group, std::to_string(scale_level), delim);
                dss.push_back(z5::filesystem::openDataset(ds_handle));
            }

            const std::string unique_id = std::to_string(std::hash<std::string>{}(dirs_group.path().string()));
            trace_data.normal3d_field = std::make_unique<Chunked3dVec3fFromUint8>(std::move(dss), scale_factor, cache, cache_root, unique_id + "_n3d");

            // Optional normal-fit diagnostics (written by vc_ngrids) to modulate loss weights.
            // Expected layout: <root>/fit_rms/0 and <root>/fit_frac_short_paths/0 (uint8, ZYX).
            try {
                z5::filesystem::handle::Group g_rms(dirs_group, "fit_rms");
                z5::filesystem::handle::Dataset ds_rms_handle(g_rms, std::to_string(scale_level), delim);
                auto ds_rms = z5::filesystem::openDataset(ds_rms_handle);

                z5::filesystem::handle::Group g_frac(dirs_group, "fit_frac_short_paths");
                z5::filesystem::handle::Dataset ds_frac_handle(g_frac, std::to_string(scale_level), delim);
                auto ds_frac = z5::filesystem::openDataset(ds_frac_handle);

                trace_data.normal3d_fit_quality = std::make_unique<NormalFitQualityWeightField>(
                    std::move(ds_rms), std::move(ds_frac), scale_factor, cache, cache_root, unique_id + "_n3d_fitq");
            } catch (const std::exception& e) {
                std::cerr << "Normal3d fit-quality fields not loaded (optional): " << e.what() << std::endl;
                trace_data.normal3d_fit_quality.reset();
            }

            std::cout << "Loaded normal3d zarr field from " << zarr_root
                      << " (ratio=" << ratio
                      << ", scale_factor=" << scale_factor
                      << ", delim='" << delim << "')" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load normal3d zarr field: " << e.what() << std::endl;
            trace_data.normal3d_field.reset();
            trace_data.normal3d_fit_quality.reset();
        }
    }

    std::unique_ptr<QuadSurface> reference_surface;
    if (params.contains("reference_surface")) {
        const nlohmann::json& ref_cfg = params["reference_surface"];
        std::string ref_path;
        if (ref_cfg.is_string()) {
            ref_path = ref_cfg.get<std::string>();
        } else if (ref_cfg.is_object()) {
            const auto path_it = ref_cfg.find("path");
            if (path_it != ref_cfg.end()) {
                if (path_it->is_string()) {
                    ref_path = path_it->get<std::string>();
                } else if (!path_it->is_null()) {
                    std::cerr << "reference_surface.path must be a string" << std::endl;
                }
            }
        }

        if (!ref_path.empty()) {
            try {
                reference_surface = load_quad_from_tifxyz(ref_path);
                loss_settings.reference_raycast.surface = reference_surface.get();
                std::cout << "Loaded reference surface from " << ref_path << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Failed to load reference surface '" << ref_path << "': " << e.what() << std::endl;
            }
        } else {
            std::cerr << "reference_surface parameter provided without a valid path" << std::endl;
        }

        if (loss_settings.reference_raycast.surface && ref_cfg.is_object()) {
            auto read_double = [&](const char* key, double current) {
                const auto it = ref_cfg.find(key);
                if (it == ref_cfg.end() || it->is_null()) {
                    return current;
                }
                if (it->is_number()) {
                    return it->get<double>();
                }
                std::cerr << "reference_surface." << key << " must be numeric" << std::endl;
                return current;
            };
            loss_settings.reference_raycast.voxel_threshold = read_double("voxel_threshold", loss_settings.reference_raycast.voxel_threshold);
            // Keep JSON key the same, but store it into the REFERENCE_RAY loss weight exclusively.
            // (This is effectively the same knob as reference_ray_weight.)
            loss_settings.w[LossType::REFERENCE_RAY] = static_cast<float>(read_double("penalty_weight", static_cast<double>(loss_settings.w[LossType::REFERENCE_RAY])));
            loss_settings.reference_raycast.sample_step     = read_double("sample_step",     loss_settings.reference_raycast.sample_step);
            loss_settings.reference_raycast.max_distance    = read_double("max_distance",    loss_settings.reference_raycast.max_distance);
            loss_settings.reference_raycast.min_clearance   = read_double("min_clearance",   loss_settings.reference_raycast.min_clearance);
            loss_settings.reference_raycast.clearance_weight = read_double("clearance_weight", loss_settings.reference_raycast.clearance_weight);
            loss_settings.reference_raycast.voxel_threshold = std::clamp(loss_settings.reference_raycast.voxel_threshold, 0.0, 255.0);
            if (loss_settings.w[LossType::REFERENCE_RAY] < 0.0f)
                loss_settings.w[LossType::REFERENCE_RAY] = 0.0f;
            if (loss_settings.reference_raycast.sample_step <= 0.0)
                loss_settings.reference_raycast.sample_step = 1.0;
            if (loss_settings.reference_raycast.min_clearance < 0.0)
                loss_settings.reference_raycast.min_clearance = 0.0;
            if (loss_settings.reference_raycast.clearance_weight < 0.0)
                loss_settings.reference_raycast.clearance_weight = 0.0;
        }

        if (loss_settings.reference_raycast_enabled()) {
            std::cout << "Reference raycast penalty enabled (threshold="
                      << loss_settings.reference_raycast.voxel_threshold
                      << ", weight=" << loss_settings.w[LossType::REFERENCE_RAY]
                      << ", step=" << loss_settings.reference_raycast.sample_step
                      << ", min_clearance=" << loss_settings.reference_raycast.min_clearance
                      << " (clear_w=" << loss_settings.reference_raycast.clearance_weight << ")";
            if (loss_settings.reference_raycast.max_distance > 0.0) {
                std::cout << ", max_distance=" << loss_settings.reference_raycast.max_distance;
            }
            std::cout << ")" << std::endl;
        }
    }
    TraceParameters trace_params;
    int snapshot_interval = params.value("snapshot-interval", 0);
    int stop_gen = params.value("generations", 100);

    // Load normal grid first if provided, so we can use its spiral-step
    std::unique_ptr<vc::core::util::NormalGridVolume> ngv;
    if (params.contains("normal_grid_path")) {
        ngv = std::make_unique<vc::core::util::NormalGridVolume>(params["normal_grid_path"].get<std::string>());
    }

    // Determine step size with priority: explicit param > normal_grid > resume_surf > default
    float step;
    if (params.contains("step_size")) {
        step = params.value("step_size", 20.0f);
    } else if (ngv) {
        // Use normal grid's spiral-step as authoritative (handles legacy surfaces with wrong scale)
        step = ngv->metadata()["spiral-step"].get<float>();
    } else if (resume_surf) {
        step = 1.0f / resume_surf->scale()[0];
    } else {
        step = 20.0f;
    }
    trace_params.unit = step*scale;

    // Validate step matches normal grid if explicit step_size was provided
    if (ngv && params.contains("step_size")) {
        float ngv_step = ngv->metadata()["spiral-step"].get<float>();
        if (std::abs(ngv_step - step) > 1e-6) {
            throw std::runtime_error("step_size parameter mismatch between normal grid volume and tracer.");
        }
    }
    trace_data.ngv = ngv.get();

    std::cout << "GrowPatch loss weights:\n"
              << "  DIST: " << loss_settings.w[LossType::DIST]
              << " STRAIGHT: " << loss_settings.w[LossType::STRAIGHT]
              << " DIRECTION: " << loss_settings.w[LossType::DIRECTION]
              << " SNAP: " << loss_settings.w[LossType::SNAP]
              << " NORMAL: " << loss_settings.w[LossType::NORMAL]
              << " NORMAL3DLINE: " << loss_settings.w[LossType::NORMAL3DLINE]
              << " REFERENCE_RAY: " << loss_settings.w[LossType::REFERENCE_RAY]
              << " SURFACE_SDT: " << loss_settings.w[LossType::SURFACE_SDT]
              << " SPACELINE: " << loss_settings.w[LossType::SPACELINE]
              << " SDIR: " << loss_settings.w[LossType::SDIR]
              << std::endl;
    int rewind_gen = params.value("rewind_gen", -1);
    loss_settings.z_min = params.value("z_min", -1);
    loss_settings.z_max = params.value("z_max", std::numeric_limits<int>::max());
    loss_settings.y_min = params.value("y_min", -1);
    loss_settings.y_max = params.value("y_max", std::numeric_limits<int>::max());
    loss_settings.x_min = params.value("x_min", -1);
    loss_settings.x_max = params.value("x_max", std::numeric_limits<int>::max());
    loss_settings.flipback_threshold = params.value("flipback_threshold", 5.0f);
    loss_settings.flipback_weight = params.value("flipback_weight", 1.0f);
    std::cout << "Anti-flipback: threshold=" << loss_settings.flipback_threshold
              << " weight=" << loss_settings.flipback_weight
              << (loss_settings.flipback_weight == 0 ? " (DISABLED)" : "") << std::endl;
    ALifeTime f_timer("empty space tracing\n");



    int w, h;
    if (resume_surf) {
        cv::Mat_<uint16_t> resume_generations = resume_surf->channel("generations");
        if (resume_generations.empty()) {
            cv::Mat_<cv::Vec3f> resume_points = resume_surf->rawPoints();
            resume_generations = cv::Mat_<uint16_t>(resume_points.size(), (uint16_t)0);

            for (int j = 0; j < resume_points.rows; ++j) {
                for (int i = 0; i < resume_points.cols; ++i) {
                    if (resume_points(j,i)[0] != -1) {
                        resume_generations(j,i) = 1;
                    }
                }
            }
            resume_surf->setChannel("generations", resume_generations);
        }
        double min_val, max_val;
        cv::minMaxLoc(resume_generations, &min_val, &max_val);
        int start_gen = (rewind_gen == -1) ? static_cast<int>(max_val) : rewind_gen;
        int gen_diff = std::max(0, stop_gen - start_gen);
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

    if (loss_settings.w[LossType::SPACELINE] > 0.0f && loss_settings.space_line_steps >= 2) {
        trace_data.space_line_compute = std::make_unique<lineLossDistance>(loss_settings.space_line_threshold,
                                                                           loss_settings.space_line_invert);
        trace_data.space_line_volume = std::make_unique<Chunked3d<uint8_t, lineLossDistance>>(
            *trace_data.space_line_compute, ds, cache, cache_root);
        std::cout << "Space-line loss EDT enabled (threshold=" << loss_settings.space_line_threshold
                  << ", steps=" << loss_settings.space_line_steps
                  << ", invert=" << loss_settings.space_line_invert << ")" << std::endl;
    }

    // Debug: test the chunk cache by reading one voxel
    passTroughComputor pass;
    Chunked3d<uint8_t,passTroughComputor> dbg_tensor(pass, ds, cache);
    trace_data.raw_volume = &dbg_tensor;
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
    cv::Mat_<cv::Vec3d> surface_normals(size, cv::Vec3d(0,0,0));  // Consistently oriented surface normals
    cv::Mat_<cv::Vec3d> reopt_anchors;
    cv::Mat_<cv::Vec3d> reopt_normals;
    cv::Mat_<uint8_t> phys_fail(size,0);
    // cv::Mat_<float> init_dist(size,0);
    cv::Mat_<uint16_t> loss_status(cv::Size(w,h),0);
    cv::Rect used_area;
    int generation = 1;

    int succ = 0;  // number of quads successfully added to the patch (each of size approx. step**2)

    int resume_pad_y = 0;
    int resume_pad_x = 0;

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

        if (params.value("vis_losses", false)) {
            cv::Mat_<float> loss_dist(generations_crop.size(), 0.0f);
            cv::Mat_<float> loss_straight(generations_crop.size(), 0.0f);
            cv::Mat_<float> loss_normal(generations_crop.size(), 0.0f);
            cv::Mat_<float> loss_snap(generations_crop.size(), 0.0f);
            cv::Mat_<float> loss_sdir(generations_crop.size(), 0.0f);

#pragma omp parallel for schedule(dynamic)
            for (int y = 0; y < generations_crop.rows; ++y) {
                for (int x = 0; x < generations_crop.cols; ++x) {
                    cv::Vec2i p = {used_area_safe.y + y, used_area_safe.x + x};
                    if (generations(p) > 0) {
                        ceres::Problem problem_dist, problem_straight, problem_normal, problem_snap, problem_sdir;
                        
                        LossSettings settings_dist, settings_straight, settings_normal, settings_snap, settings_sdir;
                        
                        settings_dist[DIST] = 1.0;
                        add_losses(problem_dist, p, trace_params, trace_data, settings_dist, LOSS_DIST);
                        
                        settings_straight[STRAIGHT] = 1.0;
                        add_losses(problem_straight, p, trace_params, trace_data, settings_straight, LOSS_STRAIGHT);
                        
                        settings_normal[NORMAL] = 1.0;
                        add_losses(problem_normal, p, trace_params, trace_data, settings_normal, LOSS_NORMALSNAP);

                        settings_snap[SNAP] = 1.0;
                        add_losses(problem_snap, p, trace_params, trace_data, settings_snap, LOSS_NORMALSNAP);

                        settings_sdir[SDIR] = 1.0;
                        add_losses(problem_sdir, p, trace_params, trace_data, settings_sdir, LOSS_SDIR);

                        double cost_dist = 0, cost_straight = 0, cost_normal = 0, cost_snap = 0, cost_sdir = 0;
                        problem_dist.Evaluate(ceres::Problem::EvaluateOptions(), &cost_dist, nullptr, nullptr, nullptr);
                        problem_straight.Evaluate(ceres::Problem::EvaluateOptions(), &cost_straight, nullptr, nullptr, nullptr);
                        problem_normal.Evaluate(ceres::Problem::EvaluateOptions(), &cost_normal, nullptr, nullptr, nullptr);
                        problem_snap.Evaluate(ceres::Problem::EvaluateOptions(), &cost_snap, nullptr, nullptr, nullptr);
                        problem_sdir.Evaluate(ceres::Problem::EvaluateOptions(), &cost_sdir, nullptr, nullptr, nullptr);

                        loss_dist(y, x) = sqrt(cost_dist);
                        loss_straight(y, x) = sqrt(cost_straight);
                        loss_normal(y, x) = sqrt(cost_normal);
                        loss_snap(y, x) = sqrt(cost_snap);
                        loss_sdir(y, x) = sqrt(cost_sdir);
                    }
                }
            }
            surf->setChannel("loss_dist", loss_dist);
            surf->setChannel("loss_straight", loss_straight);
            surf->setChannel("loss_normal", loss_normal);
            surf->setChannel("loss_snap", loss_snap);
            surf->setChannel("loss_sdir", loss_sdir);
        }

        const double area_est_vx2 = vc::surface::computeSurfaceAreaVox2(*surf);
        const double voxel_size_d = static_cast<double>(voxelsize);
        const double area_est_cm2 = area_est_vx2 * voxel_size_d * voxel_size_d / 1e8;

        surf->meta = std::make_unique<nlohmann::json>(meta_params);
        (*surf->meta)["area_vx2"] = area_est_vx2;
        (*surf->meta)["area_cm2"] = area_est_cm2;
        (*surf->meta)["max_gen"] = generation;
        (*surf->meta)["seed"] = {origin[0], origin[1], origin[2]};
        (*surf->meta)["elapsed_time_s"] = f_timer.seconds();
        if (resume_surf && !resume_surf->id.empty()) {
            (*surf->meta)["seed_surface_id"] = resume_surf->id;
            // Store grid offset for correction point remapping
            // new_coord = old_coord + offset
            const int offset_row = resume_pad_y - used_area_safe.y;
            const int offset_col = resume_pad_x - used_area_safe.x;
            (*surf->meta)["grid_offset"] = {offset_col, offset_row};
        }

        // Preserve approval and mask channels from resume surface with correct offset
        // Note: These channels are stored at raw points resolution, not scaled size
        if (resume_surf) {
            const int offset_row = resume_pad_y - used_area_safe.y;
            const int offset_col = resume_pad_x - used_area_safe.x;

            // Get raw points size (channels are stored at this resolution)
            const cv::Mat_<cv::Vec3f>* new_points = surf->rawPointsPtr();
            if (!new_points || new_points->empty()) {
                return surf;
            }
            const cv::Size raw_size = new_points->size();

            // Preserve approval channel (3-channel BGR image)
            cv::Mat old_approval = resume_surf->channel("approval", SURF_CHANNEL_NORESIZE);
            if (!old_approval.empty()) {
                // Create new approval mask matching old format at raw points resolution
                cv::Mat new_approval;
                if (old_approval.channels() == 3) {
                    new_approval = cv::Mat_<cv::Vec3b>(raw_size, cv::Vec3b(0, 0, 0));
                    for (int r = 0; r < old_approval.rows; ++r) {
                        for (int c = 0; c < old_approval.cols; ++c) {
                            int new_r = r + offset_row;
                            int new_c = c + offset_col;
                            if (new_r >= 0 && new_r < new_approval.rows &&
                                new_c >= 0 && new_c < new_approval.cols) {
                                new_approval.at<cv::Vec3b>(new_r, new_c) = old_approval.at<cv::Vec3b>(r, c);
                            }
                        }
                    }
                } else {
                    // Single channel fallback
                    new_approval = cv::Mat_<uint8_t>(raw_size, static_cast<uint8_t>(0));
                    for (int r = 0; r < old_approval.rows; ++r) {
                        for (int c = 0; c < old_approval.cols; ++c) {
                            int new_r = r + offset_row;
                            int new_c = c + offset_col;
                            if (new_r >= 0 && new_r < new_approval.rows &&
                                new_c >= 0 && new_c < new_approval.cols) {
                                new_approval.at<uint8_t>(new_r, new_c) = old_approval.at<uint8_t>(r, c);
                            }
                        }
                    }
                }
                surf->setChannel("approval", new_approval);
                std::cout << "Preserved approval mask (" << old_approval.channels() << " channels, "
                          << old_approval.cols << "x" << old_approval.rows << " -> "
                          << new_approval.cols << "x" << new_approval.rows
                          << ") with offset (" << offset_row << ", " << offset_col << ")" << std::endl;
            }

            // Preserve mask channel (single channel uint8)
            // Layer 0 of mask.tif is the validity mask - it can mask out points that have
            // valid coordinates but shouldn't be used (human corrections).
            // Strategy:
            // 1. Generate fresh validity mask from new surface points (255 if valid, 0 if -1,-1,-1)
            // 2. Overlay old mask values at correct offset (preserving human edits)
            cv::Mat old_mask = resume_surf->channel("mask", SURF_CHANNEL_NORESIZE);
            if (!old_mask.empty()) {
                // Start with validity mask based on actual point data
                cv::Mat_<uint8_t> new_mask(raw_size, static_cast<uint8_t>(0));
                for (int r = 0; r < new_points->rows; ++r) {
                    for (int c = 0; c < new_points->cols; ++c) {
                        const cv::Vec3f& p = (*new_points)(r, c);
                        if (p[0] != -1.0f) {
                            new_mask(r, c) = 255;  // Valid point
                        }
                    }
                }

                // Now overlay the old mask values (preserving human edits that masked out valid points)
                for (int r = 0; r < old_mask.rows; ++r) {
                    for (int c = 0; c < old_mask.cols; ++c) {
                        int new_r = r + offset_row;
                        int new_c = c + offset_col;
                        if (new_r >= 0 && new_r < new_mask.rows &&
                            new_c >= 0 && new_c < new_mask.cols) {
                            // Preserve the old mask value (including human edits that set 0 on valid points)
                            new_mask(new_r, new_c) = old_mask.at<uint8_t>(r, c);
                        }
                    }
                }
                surf->setChannel("mask", new_mask);
                std::cout << "Preserved mask (" << old_mask.cols << "x" << old_mask.rows << " -> "
                          << new_mask.cols << "x" << new_mask.rows
                          << ") with offset (" << offset_row << ", " << offset_col << ")" << std::endl;
            }
        }

        return surf;
    };

    cv::Vec3f vx = {1,0,0};
    cv::Vec3f vy = {0,1,0};

    // ceres::Problem big_problem;
    int loss_count = 0;
    double last_elapsed_seconds = 0.0;
    int last_succ = 0;
    int start_gen = 0;

    std::cout << "lets go! " << std::endl;

    if (resume_surf) {
        std::cout << "resuime! " << std::endl;
        float resume_step = 1.0 / resume_surf->scale()[0];
        // Only validate step match if not using normal_grid (which is authoritative for legacy surfaces)
        if (!ngv && std::abs(resume_step - step) > 1e-6) {
            throw std::runtime_error("Step size parameter mismatch between new trace and resume surface.");
        }

        cv::Mat_<cv::Vec3f> resume_points = resume_surf->rawPoints();
        cv::Mat_<uint16_t> resume_generations = resume_surf->channel("generations");

        resume_pad_x = (w - resume_points.cols) / 2;
        resume_pad_y = (h - resume_points.rows) / 2;

        used_area = cv::Rect(resume_pad_x, resume_pad_y, resume_points.cols, resume_points.rows);
 
        double min_val, max_val;
        cv::minMaxLoc(resume_generations, &min_val, &max_val);
        start_gen = (rewind_gen == -1) ? static_cast<int>(max_val) : rewind_gen;
        generation = start_gen;

        int min_gen = std::numeric_limits<int>::max();
        x0 = -1;
        y0 = -1;
        for (int j = 0; j < resume_points.rows; ++j) {
            for (int i = 0; i < resume_points.cols; ++i) {
                int target_y = resume_pad_y + j;
                int target_x = resume_pad_x + i;
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

            if (trace_data.cell_reopt_mode) {
                trace_data.boundary_mask = cv::Mat_<uchar>(trace_params.state.size(), static_cast<uchar>(0));
                trace_data.interior_mask = cv::Mat_<uchar>(trace_params.state.size(), static_cast<uchar>(0));

                bool built_from_approval = false;
                if (resume_surf) {
                    const cv::Rect resume_area(resume_pad_x, resume_pad_y, resume_points.cols, resume_points.rows);
                    cv::Mat approval = resume_surf->channel("approval", SURF_CHANNEL_NORESIZE);
                    cv::Mat_<uchar> approved = make_approved_mask(approval, resume_area, trace_params.state.size());
                    if (!approved.empty()) {
                        for (const auto& collection : trace_data.point_correction.collections()) {
                            auto seed = pick_seed_for_collection(collection, resume_area, trace_params.state.size(),
                                                                 resume_pad_x, resume_pad_y);
                            if (!seed.has_value()) {
                                continue;
                            }
                            flood_fill_unapproved(approved, *seed, trace_data.interior_mask);
                        }
                        if (cv::countNonZero(trace_data.interior_mask) > 0) {
                            compute_boundary_from_interior(trace_data.interior_mask, trace_data.boundary_mask);
                            built_from_approval = true;
                        }
                        if (built_from_approval) {
                            bool touches_border = false;
                            const int top = resume_area.y;
                            const int bottom = resume_area.br().y - 1;
                            const int left = resume_area.x;
                            const int right = resume_area.br().x - 1;

                            for (int c = left; c <= right && !touches_border; ++c) {
                                if (trace_data.interior_mask(top, c) != 0 ||
                                    trace_data.interior_mask(bottom, c) != 0) {
                                    touches_border = true;
                                }
                            }
                            for (int r = top; r <= bottom && !touches_border; ++r) {
                                if (trace_data.interior_mask(r, left) != 0 ||
                                    trace_data.interior_mask(r, right) != 0) {
                                    touches_border = true;
                                }
                            }
                            if (touches_border) {
                                trace_data.interior_mask.setTo(0);
                                trace_data.boundary_mask.setTo(0);
                                built_from_approval = false;
                                std::cout << "Cell reopt: approval interior touches resume boundary; falling back to corrections." << std::endl;
                            }
                        }
                    }
                }

                if (!built_from_approval) {
                    for (const auto& collection : trace_data.point_correction.collections()) {
                        if (collection.grid_locs_.empty()) {
                            continue;
                        }
                        if (collection.grid_locs_.size() == 1) {
                            const cv::Point center(static_cast<int>(std::round(collection.grid_locs_[0][0])),
                                                   static_cast<int>(std::round(collection.grid_locs_[0][1])));
                            const int radius = 8;
                            cv::circle(trace_data.interior_mask, center, radius, cv::Scalar(1), -1);
                            cv::circle(trace_data.boundary_mask, center, radius, cv::Scalar(1), 1);
                            continue;
                        }

                        std::vector<cv::Point> polyline;
                        polyline.reserve(collection.grid_locs_.size());
                        for (const auto& loc : collection.grid_locs_) {
                            polyline.emplace_back(static_cast<int>(std::round(loc[0])),
                                                  static_cast<int>(std::round(loc[1])));
                        }

                        if (polyline.size() >= 3) {
                            cv::fillPoly(trace_data.interior_mask, std::vector<std::vector<cv::Point>>{polyline},
                                         cv::Scalar(1), 8);
                        }

                        for (size_t i = 0; i < polyline.size(); ++i) {
                            const cv::Point& p0 = polyline[i];
                            const cv::Point& p1 = polyline[(i + 1) % polyline.size()];
                            cv::LineIterator it(trace_data.boundary_mask, p0, p1, 8);
                            for (int j = 0; j < it.count; ++j, ++it) {
                                const cv::Point pt = it.pos();
                                if (pt.y < 0 || pt.y >= trace_data.boundary_mask.rows ||
                                    pt.x < 0 || pt.x >= trace_data.boundary_mask.cols) {
                                    continue;
                                }
                                trace_data.boundary_mask(pt.y, pt.x) = 1;
                            }
                        }
                    }
                }

                const int interior_count = trace_data.interior_mask.empty()
                    ? 0
                    : cv::countNonZero(trace_data.interior_mask);
                const int boundary_count = trace_data.boundary_mask.empty()
                    ? 0
                    : cv::countNonZero(trace_data.boundary_mask);
                std::cout << "Cell reopt masks: interior=" << interior_count
                          << " boundary=" << boundary_count
                          << (built_from_approval ? " (approval)" : " (corrections)")
                          << std::endl;
            }

            std::cout << "Resuming with " << trace_data.point_correction.all_grid_locs().size() << " correction points." << std::endl;
            cv::Mat mask = resume_surf->channel("mask");
            if (!mask.empty()) {
                std::vector<std::vector<cv::Point2f>> all_hulls;
                // For single-point collections (e.g., drag-and-drop), store center and radius
                std::vector<std::pair<cv::Point2f, float>> single_point_regions;

                for (const auto& collection : trace_data.point_correction.collections()) {
                    if (collection.grid_locs_.empty()) continue;

                    if (collection.grid_locs_.size() == 1) {
                        // Single point - use a radius-based region instead of convex hull
                        // This handles the drag-and-drop case where only one correction point is set
                        cv::Point2f center(collection.grid_locs_[0][0], collection.grid_locs_[0][1]);
                        float radius = 8.0f;  // Default radius for single-point corrections
                        single_point_regions.emplace_back(center, radius);
                        std::cout << "single-point correction region at " << center << " with radius " << radius << std::endl;
                    } else {
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
                }

                for (int j = 0; j < mask.rows; ++j) {
                    for (int i = 0; i < mask.cols; ++i) {
                        if (mask.at<uint8_t>(j, i) == 0 && trace_params.state(resume_pad_y + j, resume_pad_x + i)) {
                            int target_y = resume_pad_y + j;
                            int target_x = resume_pad_x + i;
                            cv::Point2f p(target_x, target_y);
                            bool keep = false;

                            if (trace_data.cell_reopt_mode && !trace_data.interior_mask.empty()) {
                                const cv::Vec2i grid{target_y, target_x};
                                keep = point_in_bounds(trace_data.interior_mask, grid) &&
                                       trace_data.interior_mask(grid) != 0;
                            } else {
                                // Check convex hull regions
                                for (const auto& hull : all_hulls) {
                                    if (cv::pointPolygonTest(hull, p, false) >= 0) {
                                        keep = true;
                                        break;
                                    }
                                }

                                // Check single-point circular regions
                                if (!keep) {
                                    for (const auto& [center, radius] : single_point_regions) {
                                        float dx = p.x - center.x;
                                        float dy = p.y - center.y;
                                        if (dx * dx + dy * dy <= radius * radius) {
                                            keep = true;
                                            break;
                                        }
                                    }
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

            if (trace_data.cell_reopt_mode && !trace_data.interior_mask.empty()) {
                if (resample_inside_boundary(trace_params, trace_data, loss_settings)) {
                    std::cout << "Cell reopt resample completed." << std::endl;
                }

                std::vector<cv::Point> interior_points;
                cv::findNonZero(trace_data.interior_mask, interior_points);
                if (!interior_points.empty()) {
                    reopt_anchors = trace_params.dpoints.clone();
                    reopt_normals = cv::Mat_<cv::Vec3d>(trace_params.dpoints.size(), cv::Vec3d(0,0,0));
                    for (const auto& pt : interior_points) {
                        const cv::Vec2i grid{pt.y, pt.x};
                        if (trace_params.state(grid) & STATE_LOC_VALID) {
                            update_surface_normal(grid, trace_params.dpoints, trace_params.state, reopt_normals);
                        }
                    }
                    trace_data.reopt_anchors = &reopt_anchors;
                    trace_data.reopt_normals = &reopt_normals;
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
                loss_inpaint[SNAP] *= 0.0;
                loss_inpaint[DIST] *= 0.3;
                loss_inpaint[STRAIGHT] *= 0.1;
                local_optimization(radius, corr_center_i, trace_params, trace_data, loss_inpaint, false, true);
            }

            // if (!tgt_path.empty() && snapshot_interval > 0) {
            //     QuadSurface* surf = create_surface_from_state();
            //     surf->save(tgt_path.string()+"_corr_stage1", true);
            //     delete surf;
            // }

            for (const auto& opt_params : opt_centers) {
                LossSettings loss_inpaint = loss_settings;
                loss_inpaint[DIST] *= 0.3;
                loss_inpaint[STRAIGHT] *= 0.1;
                local_optimization(opt_params.radius, opt_params.center, trace_params, trace_data, loss_inpaint, false, true);
            }

            // if (!tgt_path.empty() && snapshot_interval > 0) {
            //     QuadSurface* surf = create_surface_from_state();
            //     surf->save(tgt_path.string()+"_corr_stage4", true);
            //     delete surf;
            // }

            for (const auto& opt_params : opt_centers) {
                local_optimization(opt_params.radius, opt_params.center, trace_params, trace_data, loss_settings, false, true);
            }

            // if (!tgt_path.empty() && snapshot_interval > 0) {
            //     QuadSurface* surf = create_surface_from_state();
            //     surf->save(tgt_path.string()+"_corr_stage5", true);
            //     delete surf;
            // }

            if (!tgt_path.empty() && snapshot_interval > 0) {
                QuadSurface* surf = create_surface_from_state();
                surf->save(tgt_path, true);
                delete surf;
                std::cout << "saved snapshot in " << tgt_path << std::endl;
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
        // Initialize seed normals with consistent orientation (vx cross vy = +Z direction)
        cv::Vec3d seed_normal = cv::Vec3d(vx).cross(cv::Vec3d(vy));
        seed_normal /= cv::norm(seed_normal);

        if (neural_tracer && pre_neural_gens == 0) {
            std::cout << "Initializing with neural tracer..." << std::endl;

            // Bootstrap the first quad with the neural tracer -- we already have the
            // top-left point; we construct top-right, bottom-left and bottom-right

            trace_params.dpoints(y0, x0) = origin;

            // Get hopefully-4 adjacent points; take the one with min or max z-displacement depending on required direction
            auto coordinates = neural_tracer->get_next_points({origin}, {{}}, {{}}, {{}})[0].next_u_xyzs;
            if (coordinates.empty() || cv::norm(coordinates[0]) < 1e-6) {
                std::cout << "no blobs found while bootstrapping (vertex #1, top-right)" << std::endl;
                throw std::runtime_error("Neural tracer bootstrap failed at vertex #1");
            }
            // use minimum delta-z; this choice orients the patch
            auto min_delta_z_it = std::min_element(coordinates.begin(), coordinates.end(), [&origin](const cv::Vec3f& a, const cv::Vec3f& b) {
                return std::abs(a[2] - origin[2]) < std::abs(b[2] - origin[2]);
            });
            trace_params.dpoints(y0, x0 + 1) = *min_delta_z_it;

            // Conditioned on center and right, predict below & above (choosing one arbitrarily)
            cv::Vec3f prev_v = trace_params.dpoints(y0, x0 + 1);
            coordinates = neural_tracer->get_next_points({origin}, {{}}, {prev_v}, {{}})[0].next_u_xyzs;
            if (coordinates.empty() || cv::norm(coordinates[0]) < 1e-6) {
                std::cout << "no blobs found while bootstrapping (vertex #2, bottom-left)" << std::endl;
                throw std::runtime_error("Neural tracer bootstrap failed at vertex #2");
            }
            trace_params.dpoints(y0 + 1, x0) = coordinates[0];

            // Conditioned on center (top-right of the quad!) and left and below-left, predict below
            cv::Vec3f center_xyz = trace_params.dpoints(y0, x0 + 1);
            prev_v = trace_params.dpoints(y0, x0);
            cv::Vec3f prev_diag = trace_params.dpoints(y0 + 1, x0);
            coordinates = neural_tracer->get_next_points({center_xyz}, {{}}, {prev_v}, {prev_diag})[0].next_u_xyzs;
            if (coordinates.empty() || cv::norm(coordinates[0]) < 1e-6) {
                std::cout << "no blobs found while bootstrapping (vertex #3, bottom-right)" << std::endl;
                throw std::runtime_error("Neural tracer bootstrap failed at vertex #3");
            }
            trace_params.dpoints(y0 + 1, x0 + 1) = coordinates[0];

        } else {
            // Initialise the trace at the center of the available area, as a tiny single-quad patch at the seed point
            //these are locations in the local volume!
            trace_params.dpoints(y0,x0) = origin;
            trace_params.dpoints(y0,x0+1) = origin+vx*0.1;
            trace_params.dpoints(y0+1,x0) = origin+vy*0.1;
            trace_params.dpoints(y0+1,x0+1) = origin+vx*0.1 + vy*0.1;
        }

        used_area = cv::Rect(x0,y0,2,2);

        trace_params.state(y0,x0) = STATE_LOC_VALID | STATE_COORD_VALID;
        trace_params.state(y0+1,x0) = STATE_LOC_VALID | STATE_COORD_VALID;
        trace_params.state(y0,x0+1) = STATE_LOC_VALID | STATE_COORD_VALID;
        trace_params.state(y0+1,x0+1) = STATE_LOC_VALID | STATE_COORD_VALID;

        generations(y0,x0) = 1;
        generations(y0,x0+1) = 1;
        generations(y0+1,x0) = 1;
        generations(y0+1,x0+1) = 1;

        surface_normals(y0,x0) = seed_normal;
        surface_normals(y0,x0+1) = seed_normal;
        surface_normals(y0+1,x0) = seed_normal;
        surface_normals(y0+1,x0+1) = seed_normal;

        fringe.push_back({y0,x0});
        fringe.push_back({y0+1,x0});
        fringe.push_back({y0,x0+1});
        fringe.push_back({y0+1,x0+1});
    }

    int succ_start = succ;

    // Solve the initial optimisation problem, just placing the first four vertices around the seed
    ceres::Solver::Summary big_summary;
    //just continue on resume no additional global opt
    if (!resume_surf) {
        if (!neural_tracer) {
            local_optimization(8, {y0,x0}, trace_params, trace_data, loss_settings, true);
        }
    }
    else
    {
        if (params.value("resume_opt", "skip") == "global") {
            std::cout << "global opt" << std::endl;
            local_optimization(100, {y0,x0}, trace_params, trace_data, loss_settings, false, true);
        }
        else if (params.value("resume_opt", "skip") == "local") {
            int opt_step = params.value("resume_local_opt_step", 16);
            if (opt_step <= 0) {
                std::cerr << "WARNING: resume_local_opt_step must be > 0; defaulting to 16" << std::endl;
                opt_step = 16;
            }

            int default_radius = opt_step * 2;
            int opt_radius = params.value("resume_local_opt_radius", default_radius);
            if (opt_radius <= 0) {
                std::cerr << "WARNING: resume_local_opt_radius must be > 0; defaulting to " << default_radius << std::endl;
                opt_radius = default_radius;
            }

            LocalOptimizationConfig resume_local_config;
            resume_local_config.max_iterations = params.value("resume_local_max_iters", 1000);
            if (resume_local_config.max_iterations <= 0) {
                std::cerr << "WARNING: resume_local_max_iters must be > 0; defaulting to 1000" << std::endl;
                resume_local_config.max_iterations = 1000;
            }
            resume_local_config.use_dense_qr = params.value("resume_local_dense_qr", false);

            std::cout << "local opt (step=" << opt_step
                      << ", radius=" << opt_radius
                      << ", max_iters=" << resume_local_config.max_iterations
                      << ", dense_qr=" << std::boolalpha << resume_local_config.use_dense_qr
                      << std::noboolalpha << ")" << std::endl;
            std::vector<cv::Vec2i> opt_local;
            for (int j = used_area.y; j < used_area.br().y; ++j) {
                for (int i = used_area.x; i < used_area.br().x; ++i) {
                    if ((trace_params.state(j, i) & STATE_LOC_VALID) && (i % opt_step == 0 && j % opt_step == 0)) {
                        opt_local.push_back({j, i});
                    }
                }
            }

            std::atomic<int> done = 0;
            LocalOptTimingAccumulator timing_accum;
            if (!opt_local.empty()) {
                OmpThreadPointCol opt_local_threadcol(opt_step*2+1, opt_local);
                int total = opt_local.size();
                auto start_time = std::chrono::high_resolution_clock::now();

                #pragma omp parallel
                while (true)
                {
                    cv::Vec2i p = opt_local_threadcol.next();
                    if (p[0] == -1)
                        break;

                    local_optimization(opt_radius, p, trace_params, trace_data, loss_settings, true, false, &resume_local_config, nullptr, &timing_accum);
                    done++;
#pragma omp critical
                    {
                        auto now = std::chrono::high_resolution_clock::now();
                        double elapsed_seconds = std::chrono::duration<double>(now - start_time).count();
                        double eta_seconds = (elapsed_seconds / done.load()) * (total - done.load());

                        printf("  optimizing... %d/%d (%.2f%%) | elapsed: %.1fs | eta: %.1fs\r",
                               done.load(), total, (100.0 * done.load() / total), elapsed_seconds, eta_seconds);
                        fflush(stdout);
                    }
                }
                printf("\n");
                auto end_time = std::chrono::high_resolution_clock::now();
                double wall_time = std::chrono::duration<double>(end_time - start_time).count();
                timing_accum.print(wall_time);
            }
        }
        else if (params.value("inpaint", false)) {
            cv::Mat mask = resume_surf->channel("mask");
            cv::Mat_<uchar> hole_mask(trace_params.state.size(), (uchar)0);

            cv::Mat active_area_mask(trace_params.state.size(), (uchar)0);
            for (int y = 0; y < trace_params.state.rows; ++y) {
                for (int x = 0; x < trace_params.state.cols; ++x) {
                    if (trace_params.state(y, x) & STATE_LOC_VALID) {
                        active_area_mask.at<uchar>(y, x) = 255;
                    }
                }
            }

            if (!mask.empty()) {
                cv::Mat padded_mask = cv::Mat::zeros(trace_params.state.size(), CV_8U);
                mask.copyTo(padded_mask(used_area));
                cv::bitwise_and(active_area_mask, padded_mask, hole_mask);
            } else {
                active_area_mask.copyTo(hole_mask);
            }

            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;
            cv::findContours(hole_mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

            std::cout << "performing inpaint on " << contours.size() << " potential holes" << std::endl;

            int inpaint_count = 0;
            int inpaint_skip = 0;

            // cv::Mat_<cv::Vec3b> vis(hole_mask.size());

    #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < contours.size(); i++) {
                if (hierarchy[i][3] != -1) { // It's a hole
                    cv::Rect roi = cv::boundingRect(contours[i]);

                    int margin = 4;
                    roi.x = std::max(0, roi.x - margin);
                    roi.y = std::max(0, roi.y - margin);
                    roi.width = std::min(hole_mask.cols - roi.x, roi.width + 2 * margin);
                    roi.height = std::min(hole_mask.rows - roi.y, roi.height + 2 * margin);

                    bool insufficient_border =
                        roi.width <= 4 || roi.height <= 4 ||
                        roi.x <= 1 || roi.y <= 1 ||
                        (roi.x + roi.width) > hole_mask.cols - 2 ||
                        (roi.y + roi.height) > hole_mask.rows - 2;
                    if (insufficient_border) {
    #pragma omp atomic
                        inpaint_skip++;
    #pragma omp critical
                        {
                            std::cout << "skip inpaint: insufficient margin around roi " << roi << std::endl;
                        }
                        continue;
                    }

                    // std::cout << hole_mask.size() << trace_params.state.size() << resume_pad_x << "x" << resume_pad_y << std::endl;

                    // cv::Point testp(2492+resume_pad_x, 508+resume_pad_y);
                    // cv::Point testp(2500+resume_pad_x, 566+resume_pad_y);
                    // cv::Point testp(2340+resume_pad_x, 577+resume_pad_y);

                    // cv::rectangle(vis, roi, cv::Scalar(255,255,255));

                    // if (!roi.contains(testp)) {
                    //     // std::cout << "skip " << roi << std::endl;
                    //     continue;
                    // }

                    cv::Mat_<uchar> inpaint_mask(roi.size(), (uchar)1);

                    std::vector<cv::Point> hole_contour_roi;
                    for(const auto& p : contours[i]) {
                        hole_contour_roi.push_back({p.x - roi.x, p.y - roi.y});
                    }
                    std::vector<std::vector<cv::Point>> contours_to_fill = {hole_contour_roi};
                    cv::fillPoly(inpaint_mask, contours_to_fill, cv::Scalar(0));

                    // std::cout << "Inpainting hole at " << roi << " - " << inpaint_count << "+" << inpaint_skip << "/" << contours.size() << std::endl;
                    bool did_inpaint = false;
                    try {
                        did_inpaint = inpaint(roi, inpaint_mask, trace_params, trace_data);
                    } catch (const cv::Exception& ex) {
    #pragma omp atomic
                        inpaint_skip++;
    #pragma omp critical
                        {
                            std::cout << "skip inpaint: OpenCV exception for roi " << roi << " => " << ex.what() << std::endl;
                        }
                        continue;
                    } catch (const std::exception& ex) {
    #pragma omp atomic
                        inpaint_skip++;
    #pragma omp critical
                        {
                            std::cout << "skip inpaint: exception for roi " << roi << " => " << ex.what() << std::endl;
                        }
                        continue;
                    } catch (...) {
    #pragma omp atomic
                        inpaint_skip++;
    #pragma omp critical
                        {
                            std::cout << "skip inpaint: unknown exception for roi " << roi << std::endl;
                        }
                        continue;
                    }

                    if (!did_inpaint) {
    #pragma omp atomic
                        inpaint_skip++;
    #pragma omp critical
                        {
                            std::cout << "skip inpaint: mask border check failed for roi " << roi << std::endl;
                        }
                        continue;
                    }

    #pragma omp critical
                    {
                        if (snapshot_interval > 0 && !tgt_path.empty() && inpaint_count % snapshot_interval == 0) {
                            QuadSurface* surf = create_surface_from_state();
                            surf->save(tgt_path, true);
                            delete surf;
                            std::cout << "saved snapshot in " << tgt_path << " (" << inpaint_count << "+" << inpaint_skip << "/" << contours.size() << ")" << std::endl;
                        }
                    }

    #pragma omp atomic
                    inpaint_count++;
                }
                else
    #pragma omp atomic
                    inpaint_skip++;
            }

            // cv::imwrite("vis_inp_rect.tif", vis);
        }
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
        bool global_opt = generation <= 10 && !resume_surf;

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

        if (neural_tracer && generation > pre_neural_gens) {
            std::unordered_set<cv::Vec2i> cands_processed;  // subset of cands we've already passed to the neural tracer in this gen
            while (true) {

                float const min_pair_distance = 4.f;  // points closer than this in uv-space aren't processed together

                std::vector<cv::Vec2i> batch_cands;
                batch_cands.reserve(neural_batch_size);
                for (auto const& p : cands) {
                    if (trace_params.state(p) & (STATE_LOC_VALID | STATE_COORD_VALID))
                        continue;
                    if (cands_processed.contains(p))
                        continue;
                    if (min_dist(p, batch_cands) < min_pair_distance)
                        continue;
                    // TODO: also skip if its neighbors are outside the z-range (c.f. ceres case checking avg)
                    batch_cands.push_back(p);
                    cands_processed.insert(p);
                    if (batch_cands.size() == neural_batch_size)
                        break;
                }

                auto const points_placed = call_neural_tracer_for_points(batch_cands, trace_params, neural_tracer.get());

                for (cv::Vec2i const &p : points_placed) {
                    generations(p) = generation;
                    if (!used_area.contains(cv::Point(p[1],p[0]))) {
                        used_area = used_area | cv::Rect(p[1],p[0],1,1);
                    }
                    fringe.push_back(p);
                    succ_gen_ps.push_back(p);
                }
                succ += points_placed.size();
                succ_gen += points_placed.size();

                if (cands_processed.size() == cands.size())
                    break;

            }  // end loop over batches

        } else {

            // Snapshot positions before per-point optimization for flip detection
            cv::Mat_<cv::Vec3d> positions_before_perpoint = trace_params.dpoints.clone();

            // Configure anti-flipback constraint for per-point optimization
            // Note: new points won't have surface normals yet, but the loss function handles this
            AntiFlipbackConfig perpoint_flipback_config;
            perpoint_flipback_config.anchors = &positions_before_perpoint;
            perpoint_flipback_config.surface_normals = &surface_normals;
            perpoint_flipback_config.threshold = loss_settings.flipback_threshold;
            perpoint_flipback_config.weight = loss_settings.flipback_weight;

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

                    avg /= ref_count;

                    //"fast" skip based on avg xyz value out of limits
                    if (avg[2] < loss_settings.z_min || avg[2] > loss_settings.z_max ||
                        avg[1] < loss_settings.y_min || avg[1] > loss_settings.y_max ||
                        avg[0] < loss_settings.x_min || avg[0] > loss_settings.x_max)
                        continue;



                    cv::Vec3d init = trace_params.dpoints(best_l) + random_perturbation();
                    trace_params.dpoints(p) = init;

                    ceres::Problem problem;
                    trace_params.state(p) = STATE_LOC_VALID | STATE_COORD_VALID;

                    int flags = LOSS_DIST | LOSS_STRAIGHT;
                    if (trace_data.normal3d_field)
                        flags | LOSS_3DNORMALLINE;
                    if (trace_data.space_line_volume && loss_settings.w[LossType::SPACELINE] > 0.0f)
                        flags |= LOSS_SPACELINE;

                    add_losses(problem, p, trace_params, trace_data, loss_settings, flags);

                    std::vector<double*> parameter_blocks;
                    problem.GetParameterBlocks(&parameter_blocks);
                    for (auto& block : parameter_blocks) {
                        problem.SetParameterBlockConstant(block);
                    }
                    problem.SetParameterBlockVariable(&trace_params.dpoints(p)[0]);

                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);

                    local_optimization(1, p, trace_params, trace_data, loss_settings, true, false, nullptr, &perpoint_flipback_config);
                    if (local_opt_r > 1)
                        local_optimization(local_opt_r, p, trace_params, trace_data, loss_settings, true, false, nullptr, &perpoint_flipback_config);

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
                }  // end parallel iteration over cands
            }
        }

        // Update surface normals for all newly added points and their neighbors
        // This must be done after parallel section completes
        for (const auto& p : succ_gen_ps) {
            update_surface_normal(p, trace_params.dpoints, trace_params.state, surface_normals);
            // Also update neighbors since their geometry may have changed
            static const cv::Vec2i neighbor_offsets[] = {{-1,0}, {1,0}, {0,-1}, {0,1}};
            for (const auto& off : neighbor_offsets) {
                cv::Vec2i neighbor = p + off;
                if (neighbor[0] >= 0 && neighbor[0] < trace_params.dpoints.rows &&
                    neighbor[1] >= 0 && neighbor[1] < trace_params.dpoints.cols &&
                    (trace_params.state(neighbor) & STATE_LOC_VALID)) {
                    update_surface_normal(neighbor, trace_params.dpoints, trace_params.state, surface_normals);
                }
            }
        }

        if (neural_tracer && generation > pre_neural_gens) {
            // Skip optimizations
        } else if (!global_opt) {
            // For late generations, instead of re-solving the global problem, solve many local-ish problems, around each
            // of the newly added points
            std::vector<cv::Vec2i> opt_local;
            for(auto p : succ_gen_ps)
                if (p[0] % 4 == 0 && p[1] % 4 == 0)
                    opt_local.push_back(p);

            int done = 0;

            if (!opt_local.empty()) {
                // Snapshot positions before optimization for flip detection
                cv::Mat_<cv::Vec3d> positions_before_opt = trace_params.dpoints.clone();

                // Configure anti-flipback constraint
                AntiFlipbackConfig flipback_config;
                flipback_config.anchors = &positions_before_opt;
                flipback_config.surface_normals = &surface_normals;
                flipback_config.threshold = loss_settings.flipback_threshold;
                flipback_config.weight = loss_settings.flipback_weight;

                OmpThreadPointCol opt_local_threadcol(17, opt_local);

#pragma omp parallel
                while (true)
                {
                    CachedChunked3dInterpolator<uint8_t,thresholdedDistance> interp(proc_tensor);
                    cv::Vec2i p = opt_local_threadcol.next();
                    if (p[0] == -1)
                        break;

                    local_optimization(8, p, trace_params, trace_data, loss_settings, true, false, nullptr, &flipback_config);

#pragma omp atomic
                    done++;
                }
            }
        }
        else {
            //we do the global opt only every 8 gens, as every add does a small local solve anyweays
            if (generation % 8 == 0) {
                // Snapshot positions before global optimization
                cv::Mat_<cv::Vec3d> positions_before_opt = trace_params.dpoints.clone();

                // Configure anti-flipback constraint
                AntiFlipbackConfig flipback_config;
                flipback_config.anchors = &positions_before_opt;
                flipback_config.surface_normals = &surface_normals;
                flipback_config.threshold = loss_settings.flipback_threshold;
                flipback_config.weight = loss_settings.flipback_weight;

                local_optimization(stop_gen+10, {y0,x0}, trace_params, trace_data, loss_settings, false, true, nullptr, &flipback_config);
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
