#include <iostream>
#include "utils/Json.hpp"

#include "vc/core/types/VcDataset.hpp"

#include <algorithm>
#include <boost/program_options.hpp>
#include <cctype>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>
#include <random>
#include <set>

#include "vc/core/util/PointIndex.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/SurfaceArea.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vc/core/util/DateTime.hpp"
#include "vc/core/util/StreamOperators.hpp"
#include "vc/tracer/Tracer.hpp"


using shape = std::vector<size_t>;


using Json = utils::Json;
namespace po = boost::program_options;

namespace {

constexpr const char* kAutoGrowPrefix = "auto_grown_";

struct SweepDimension {
    std::string key;
    std::vector<nlohmann::json> values;
};

struct SweepRun {
    nlohmann::json params;
    nlohmann::json values;
};

struct DistanceStats {
    int samples = 0;
    int within_tolerance = 0;
    double mean = 0.0;
    double rms = 0.0;
    double p95 = 0.0;
    double max = 0.0;
};

struct SweepMetrics {
    bool has_result = false;
    int valid_points = 0;
    int valid_quads = 0;
    double area_vx2 = 0.0;
    double area_cm2 = 0.0;
    double area_ratio = 0.0;
    double area_abs_error_ratio = 1.0;
    DistanceStats result_to_target;
    DistanceStats target_to_result;
    double result_coverage_fraction = 0.0;
    double target_coverage_fraction = 0.0;
    double coverage_adjusted_mean_distance_vx = std::numeric_limits<double>::infinity();
    int valid_bbox_pixels = 0;
    int enclosed_hole_pixels = 0;
    double valid_bbox_fill_fraction = 0.0;
    double enclosed_hole_fraction = 0.0;
    double mesh_completeness_score = 0.0;
    double objective_score = std::numeric_limits<double>::infinity();
};

struct SeedCoordinate {
    cv::Vec3f point{0.0f, 0.0f, 0.0f};
    std::string label;
};

static std::string zero_padded_index(size_t idx, int width = 4)
{
    std::ostringstream os;
    os << std::setw(width) << std::setfill('0') << idx;
    return os.str();
}

static std::string seed_label_from_point(const cv::Vec3f& p)
{
    std::ostringstream os;
    os << std::fixed << std::setprecision(3)
       << "seed_" << p[0] << "_" << p[1] << "_" << p[2];
    std::string label = os.str();
    for (char& c : label) {
        if (c == '-') {
            c = 'm';
        } else if (c == '.') {
            c = 'p';
        } else if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
            c = '_';
        }
    }
    return label;
}

static SeedCoordinate parse_seed_coordinate_string(std::string value)
{
    std::replace(value.begin(), value.end(), ',', ' ');
    std::istringstream is(value);
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    std::string trailing;
    if (!(is >> x >> y >> z) || (is >> trailing)) {
        throw std::runtime_error("seed coordinate must contain exactly three numbers: x,y,z");
    }
    cv::Vec3f point(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
    return {point, seed_label_from_point(point)};
}

static SeedCoordinate parse_seed_coordinate_json(const nlohmann::json& json)
{
    cv::Vec3f point;
    if (json.is_array() && json.size() == 3) {
        point = {
            json.at(0).get<float>(),
            json.at(1).get<float>(),
            json.at(2).get<float>()
        };
    } else if (json.is_object() && json.contains("x") && json.contains("y") && json.contains("z")) {
        point = {
            json.at("x").get<float>(),
            json.at("y").get<float>(),
            json.at("z").get<float>()
        };
    } else {
        throw std::runtime_error("seed coordinate JSON entries must be [x,y,z] or {\"x\":...,\"y\":...,\"z\":...}");
    }
    return {point, seed_label_from_point(point)};
}

static std::vector<SeedCoordinate> parse_vc3d_point_collections_json(const nlohmann::json& json)
{
    if (!json.is_object() || !json.contains("collections")) {
        return {};
    }

    struct IndexedSeed {
        uint64_t collection_id = 0;
        uint64_t point_id = 0;
        int64_t creation_time = 0;
        size_t insertion_order = 0;
        SeedCoordinate seed;
    };

    std::vector<IndexedSeed> indexed;
    const auto& collections = json.at("collections");
    size_t insertion_order = 0;

    auto add_point = [&](uint64_t collection_id,
                         uint64_t point_id,
                         int64_t creation_time,
                         const nlohmann::json& point_json) {
        const nlohmann::json* coord_json = nullptr;
        if (point_json.is_object()) {
            if (point_json.contains("p")) {
                coord_json = &point_json.at("p");
            } else if (point_json.contains("position")) {
                coord_json = &point_json.at("position");
            } else if (point_json.contains("x") && point_json.contains("y") && point_json.contains("z")) {
                coord_json = &point_json;
            }
        } else if (point_json.is_array()) {
            coord_json = &point_json;
        }

        if (!coord_json) {
            return;
        }

        SeedCoordinate seed = parse_seed_coordinate_json(*coord_json);
        std::ostringstream label;
        label << "seed_col_" << collection_id
              << "_pt_" << point_id
              << "_" << seed.label;
        seed.label = label.str();
        indexed.push_back({collection_id, point_id, creation_time, insertion_order++, std::move(seed)});
    };

    if (collections.is_object()) {
        for (auto collection_it = collections.begin(); collection_it != collections.end(); ++collection_it) {
            uint64_t collection_id = 0;
            try {
                collection_id = std::stoull(collection_it.key());
            } catch (...) {
                continue;
            }

            const auto& collection_json = collection_it.value();
            if (!collection_json.is_object() || !collection_json.contains("points")) {
                continue;
            }

            const auto& points = collection_json.at("points");
            if (!points.is_object()) {
                continue;
            }
            for (auto point_it = points.begin(); point_it != points.end(); ++point_it) {
                uint64_t point_id = 0;
                try {
                    point_id = std::stoull(point_it.key());
                } catch (...) {
                    continue;
                }
                const auto& point_json = point_it.value();
                const int64_t creation_time = point_json.is_object() && point_json.contains("creation_time")
                    ? point_json.at("creation_time").get<int64_t>()
                    : 0;
                add_point(collection_id, point_id, creation_time, point_json);
            }
        }
    } else if (collections.is_array()) {
        for (const auto& collection_json : collections) {
            if (!collection_json.is_object() || !collection_json.contains("points")) {
                continue;
            }
            const uint64_t collection_id = collection_json.value("id", static_cast<uint64_t>(0));
            const auto& points = collection_json.at("points");
            if (!points.is_array()) {
                continue;
            }
            for (const auto& point_json : points) {
                if (!point_json.is_object()) {
                    add_point(collection_id, 0, 0, point_json);
                    continue;
                }
                const uint64_t point_id = point_json.value("id", static_cast<uint64_t>(0));
                const int64_t creation_time = point_json.value("creation_time", static_cast<int64_t>(0));
                add_point(collection_id, point_id, creation_time, point_json);
            }
        }
    }

    std::sort(indexed.begin(), indexed.end(),
              [](const IndexedSeed& a, const IndexedSeed& b) {
                  if (a.collection_id != b.collection_id) {
                      return a.collection_id < b.collection_id;
                  }
                  if (a.creation_time != b.creation_time) {
                      return a.creation_time < b.creation_time;
                  }
                  if (a.point_id != b.point_id) {
                      return a.point_id < b.point_id;
                  }
                  return a.insertion_order < b.insertion_order;
              });

    std::vector<SeedCoordinate> coords;
    coords.reserve(indexed.size());
    for (auto& item : indexed) {
        coords.push_back(std::move(item.seed));
    }
    return coords;
}

static std::vector<SeedCoordinate> parse_seed_coordinates_file(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open seed coordinates file " + path.string());
    }

    std::ostringstream buffer;
    buffer << in.rdbuf();
    const std::string content = buffer.str();

    nlohmann::json json = nlohmann::json::parse(content, nullptr, false);
    if (!json.is_discarded()) {
        if (auto coords = parse_vc3d_point_collections_json(json); !coords.empty()) {
            return coords;
        }
        if (!json.is_array()) {
            throw std::runtime_error("seed coordinates JSON file must contain an array or VC3D point collections object");
        }
        std::vector<SeedCoordinate> coords;
        coords.reserve(json.size());
        for (const auto& item : json) {
            coords.push_back(parse_seed_coordinate_json(item));
        }
        return coords;
    }

    std::vector<SeedCoordinate> coords;
    std::istringstream lines(content);
    std::string line;
    while (std::getline(lines, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) {
            line.resize(comment);
        }
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }
        coords.push_back(parse_seed_coordinate_string(line));
    }
    return coords;
}

static QuadSurface* find_source_surface_containing_point(const std::vector<QuadSurface*>& surfaces,
                                                         const cv::Vec3f& point)
{
    for (QuadSurface* surface : surfaces) {
        if (surface && contains(*surface, point)) {
            return surface;
        }
    }
    return nullptr;
}

static nlohmann::json json_from_utils(const utils::Json& json)
{
    return nlohmann::json::parse(json.dump());
}

static utils::Json json_to_utils(const nlohmann::json& json)
{
    return utils::Json::parse(json.dump());
}

static double json_number_or(const nlohmann::json& json, const char* key, double fallback)
{
    if (!json.contains(key) || !json.at(key).is_number()) {
        return fallback;
    }
    return json.at(key).get<double>();
}

static std::vector<nlohmann::json> expand_sweep_values(const std::string& key, const nlohmann::json& spec)
{
    if (spec.is_array()) {
        return spec.get<std::vector<nlohmann::json>>();
    }

    if (spec.is_object() && spec.contains("values")) {
        const auto& values = spec.at("values");
        if (!values.is_array()) {
            throw std::runtime_error("sweep key '" + key + "' has non-array values");
        }
        return values.get<std::vector<nlohmann::json>>();
    }

    if (spec.is_object()) {
        const bool has_minmax = spec.contains("min") && spec.contains("max");
        const bool has_startend = spec.contains("start") && spec.contains("end");
        if ((has_minmax || has_startend) && spec.contains("step")) {
            const double lo = has_minmax ? spec.at("min").get<double>() : spec.at("start").get<double>();
            const double hi = has_minmax ? spec.at("max").get<double>() : spec.at("end").get<double>();
            const double step = spec.at("step").get<double>();
            if (!std::isfinite(lo) || !std::isfinite(hi) || !std::isfinite(step) || step == 0.0) {
                throw std::runtime_error("sweep key '" + key + "' has invalid numeric range");
            }

            std::vector<nlohmann::json> values;
            const int direction = step > 0.0 ? 1 : -1;
            for (double v = lo; direction > 0 ? v <= hi + std::abs(step) * 1e-6 : v >= hi - std::abs(step) * 1e-6; v += step) {
                values.push_back(v);
                if (values.size() > 100000) {
                    throw std::runtime_error("sweep key '" + key + "' produced too many values");
                }
            }
            return values;
        }
    }

    throw std::runtime_error("sweep key '" + key + "' must be an array, {values:[...]}, or {min,max,step}");
}

static std::vector<SweepRun> expand_sweep_runs(const nlohmann::json& base_params, const std::filesystem::path& sweep_ranges_path)
{
    const nlohmann::json ranges = nlohmann::json::parse(std::ifstream(sweep_ranges_path));
    if (!ranges.is_object()) {
        throw std::runtime_error("sweep ranges JSON must be an object");
    }

    std::vector<SweepDimension> dims;
    for (auto it = ranges.begin(); it != ranges.end(); ++it) {
        auto values = expand_sweep_values(it.key(), it.value());
        if (values.empty()) {
            throw std::runtime_error("sweep key '" + it.key() + "' has no values");
        }
        dims.push_back({it.key(), std::move(values)});
    }

    if (dims.empty()) {
        throw std::runtime_error("sweep ranges JSON did not contain any parameters");
    }

    std::vector<SweepRun> runs;
    nlohmann::json current_values = nlohmann::json::object();

    std::function<void(size_t, nlohmann::json)> rec = [&](size_t dim, nlohmann::json params) {
        if (dim == dims.size()) {
            runs.push_back({std::move(params), current_values});
            return;
        }

        for (const auto& value : dims[dim].values) {
            nlohmann::json next = params;
            next[dims[dim].key] = value;
            current_values[dims[dim].key] = value;
            rec(dim + 1, std::move(next));
        }
        current_values.erase(dims[dim].key);
    };

    rec(0, base_params);
    return runs;
}

static std::vector<SweepRun> sample_sweep_runs_latin_hypercube(const nlohmann::json& base_params,
                                                               const std::filesystem::path& sweep_ranges_path,
                                                               int max_runs,
                                                               uint32_t seed)
{
    const nlohmann::json ranges = nlohmann::json::parse(std::ifstream(sweep_ranges_path));
    if (!ranges.is_object()) {
        throw std::runtime_error("sweep ranges JSON must be an object");
    }
    if (max_runs <= 0) {
        throw std::runtime_error("--sweep-max-runs must be positive for lhs sampling");
    }

    std::vector<SweepDimension> dims;
    for (auto it = ranges.begin(); it != ranges.end(); ++it) {
        auto values = expand_sweep_values(it.key(), it.value());
        if (values.empty()) {
            throw std::runtime_error("sweep key '" + it.key() + "' has no values");
        }
        dims.push_back({it.key(), std::move(values)});
    }
    if (dims.empty()) {
        throw std::runtime_error("sweep ranges JSON did not contain any parameters");
    }

    std::mt19937 rng(seed);
    std::vector<std::vector<size_t>> dim_indices;
    dim_indices.reserve(dims.size());
    for (const auto& dim : dims) {
        std::vector<size_t> indices;
        indices.reserve(static_cast<size_t>(max_runs));
        const size_t n = dim.values.size();
        for (int i = 0; i < max_runs; ++i) {
            const double t = (static_cast<double>(i) + 0.5) / static_cast<double>(max_runs);
            indices.push_back(std::min(n - 1, static_cast<size_t>(std::floor(t * static_cast<double>(n)))));
        }
        std::shuffle(indices.begin(), indices.end(), rng);
        dim_indices.push_back(std::move(indices));
    }

    std::vector<SweepRun> runs;
    runs.reserve(static_cast<size_t>(max_runs));
    std::set<std::string> seen;
    for (int run = 0; run < max_runs; ++run) {
        nlohmann::json params = base_params;
        nlohmann::json values = nlohmann::json::object();
        for (size_t dim_idx = 0; dim_idx < dims.size(); ++dim_idx) {
            const auto& dim = dims[dim_idx];
            const nlohmann::json& value = dim.values[dim_indices[dim_idx][run]];
            params[dim.key] = value;
            values[dim.key] = value;
        }

        const std::string key = values.dump();
        if (seen.insert(key).second) {
            runs.push_back({std::move(params), std::move(values)});
        }
    }

    return runs;
}

static double median_surface_step_vx(const QuadSurface& surface)
{
    std::vector<double> steps;
    const auto* points = surface.rawPointsPtr();
    if (!points || points->empty()) {
        return 1.0;
    }

    steps.reserve(static_cast<size_t>(points->rows + points->cols));
    for (int y = 0; y < points->rows; ++y) {
        for (int x = 0; x + 1 < points->cols; ++x) {
            const cv::Vec3f& a = (*points)(y, x);
            const cv::Vec3f& b = (*points)(y, x + 1);
            if (a[0] != -1.0f && b[0] != -1.0f) {
                const double d = cv::norm(b - a);
                if (std::isfinite(d) && d > 0.0) {
                    steps.push_back(d);
                }
            }
        }
    }
    for (int y = 0; y + 1 < points->rows; ++y) {
        for (int x = 0; x < points->cols; ++x) {
            const cv::Vec3f& a = (*points)(y, x);
            const cv::Vec3f& b = (*points)(y + 1, x);
            if (a[0] != -1.0f && b[0] != -1.0f) {
                const double d = cv::norm(b - a);
                if (std::isfinite(d) && d > 0.0) {
                    steps.push_back(d);
                }
            }
        }
    }

    if (steps.empty()) {
        return 1.0;
    }
    auto mid = steps.begin() + static_cast<std::ptrdiff_t>(steps.size() / 2);
    std::nth_element(steps.begin(), mid, steps.end());
    return *mid;
}

static cv::Point find_center_seed_point(const cv::Mat_<cv::Vec3f>& points)
{
    const cv::Point center(points.cols / 2, points.rows / 2);
    if (points(center.y, center.x)[0] != -1.0f) {
        return center;
    }

    cv::Point best = center;
    int best_dist2 = std::numeric_limits<int>::max();
    for (int y = 0; y < points.rows; ++y) {
        for (int x = 0; x < points.cols; ++x) {
            if (points(y, x)[0] == -1.0f) {
                continue;
            }
            const int dx = x - center.x;
            const int dy = y - center.y;
            const int dist2 = dx * dx + dy * dy;
            if (dist2 < best_dist2) {
                best_dist2 = dist2;
                best = {x, y};
            }
        }
    }
    return best;
}

static std::unique_ptr<QuadSurface> load_or_create_center_cutout(QuadSurface& target,
                                                                 const std::filesystem::path& target_path,
                                                                 double cutout_cm,
                                                                 double voxelsize)
{
    const auto* target_points = target.rawPointsPtr();
    if (!target_points || target_points->empty()) {
        throw std::runtime_error("sweep target tifxyz is empty");
    }

    const double median_step_vx = std::max(1e-6, median_surface_step_vx(target));
    const double step_cm = std::isfinite(voxelsize) && voxelsize > 0.0
        ? median_step_vx * voxelsize / 10000.0
        : 0.0;
    int cutout_px = step_cm > 0.0
        ? static_cast<int>(std::lround(cutout_cm / step_cm))
        : std::min(target_points->cols, target_points->rows) / 4;
    cutout_px = std::max(3, cutout_px);
    cutout_px = std::min(cutout_px, std::min(target_points->cols, target_points->rows));

    const cv::Point seed = find_center_seed_point(*target_points);
    const int x0 = std::clamp(seed.x - cutout_px / 2, 0, std::max(0, target_points->cols - cutout_px));
    const int y0 = std::clamp(seed.y - cutout_px / 2, 0, std::max(0, target_points->rows - cutout_px));
    const cv::Rect roi(x0, y0, cutout_px, cutout_px);

    std::filesystem::path normalized_target = target_path;
    while (!normalized_target.empty() && normalized_target.filename().empty()) {
        normalized_target = normalized_target.parent_path();
    }
    const std::filesystem::path cutout_dir =
        normalized_target.parent_path() /
        ("sweep_seed_cutout_" + normalized_target.filename().string() + "_" + std::to_string(cutout_px) + "px");
    if (std::filesystem::exists(cutout_dir / "meta.json") &&
        std::filesystem::exists(cutout_dir / "x.tif") &&
        std::filesystem::exists(cutout_dir / "y.tif") &&
        std::filesystem::exists(cutout_dir / "z.tif")) {
        std::cout << "Reusing sweep seed cutout " << cutout_dir << std::endl;
        return std::make_unique<QuadSurface>(cutout_dir);
    }

    cv::Mat_<cv::Vec3f> cutout = (*target_points)(roi).clone();
    auto seed_surface = std::make_unique<QuadSurface>(cutout, target.scale());
    seed_surface->meta = target.meta;
    seed_surface->meta["source"] = "vc_grow_seg_from_segments_sweep_cutout";
    seed_surface->meta["sweep_target"] = normalized_target.string();
    seed_surface->meta["sweep_cutout_cm"] = cutout_cm;
    seed_surface->meta["sweep_cutout_px"] = cutout_px;
    seed_surface->meta["sweep_cutout_x"] = x0;
    seed_surface->meta["sweep_cutout_y"] = y0;
    const std::string uuid = cutout_dir.filename().string();
    seed_surface->save(cutout_dir, uuid, true);
    std::cout << "Created sweep seed cutout " << cutout_dir
              << " roi=" << roi
              << " edge_cm=" << cutout_cm << std::endl;
    return std::make_unique<QuadSurface>(cutout_dir);
}

static DistanceStats sampled_distance_stats(const cv::Mat_<cv::Vec3f>& samples,
                                            const PointIndex& index,
                                            int valid_count,
                                            int max_samples,
                                            double tolerance)
{
    DistanceStats stats;
    if (samples.empty() || index.empty() || valid_count <= 0 || max_samples <= 0) {
        return stats;
    }

    const int stride = std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(valid_count) / max_samples))));
    std::vector<double> distances;
    distances.reserve(static_cast<size_t>(std::min(valid_count, max_samples)));

    for (auto [row, col, point] : ValidPointRange<const cv::Vec3f>(&samples)) {
        if ((row % stride) != 0 || (col % stride) != 0) {
            continue;
        }
        auto nearest = index.nearest(point);
        if (!nearest) {
            continue;
        }
        const double d = std::sqrt(static_cast<double>(nearest->distanceSq));
        if (!std::isfinite(d)) {
            continue;
        }
        distances.push_back(d);
        if (d <= tolerance) {
            stats.within_tolerance++;
        }
    }

    if (distances.empty()) {
        return stats;
    }

    stats.samples = static_cast<int>(distances.size());
    double sum = 0.0;
    double sum_sq = 0.0;
    for (double d : distances) {
        sum += d;
        sum_sq += d * d;
        stats.max = std::max(stats.max, d);
    }
    stats.mean = sum / distances.size();
    stats.rms = std::sqrt(sum_sq / distances.size());
    std::sort(distances.begin(), distances.end());
    const size_t p95_idx = std::min(distances.size() - 1, static_cast<size_t>(std::floor(0.95 * (distances.size() - 1))));
    stats.p95 = distances[p95_idx];
    return stats;
}

static nlohmann::json distance_stats_json(const DistanceStats& stats)
{
    return {
        {"samples", stats.samples},
        {"within_tolerance", stats.within_tolerance},
        {"mean_vx", stats.mean},
        {"rms_vx", stats.rms},
        {"p95_vx", stats.p95},
        {"max_vx", stats.max}
    };
}

static void add_completeness_metrics(QuadSurface& result, SweepMetrics& metrics)
{
    cv::Mat valid = result.validMask();
    if (valid.empty()) {
        return;
    }

    std::vector<cv::Point> valid_points;
    cv::findNonZero(valid, valid_points);
    if (valid_points.empty()) {
        return;
    }

    const cv::Rect bbox = cv::boundingRect(valid_points);
    const cv::Mat valid_roi = valid(bbox);
    metrics.valid_bbox_pixels = bbox.area();
    const int valid_pixels = cv::countNonZero(valid_roi);
    metrics.valid_bbox_fill_fraction = metrics.valid_bbox_pixels > 0
        ? static_cast<double>(valid_pixels) / metrics.valid_bbox_pixels
        : 0.0;

    cv::Mat invalid_roi;
    cv::compare(valid_roi, 0, invalid_roi, cv::CMP_EQ);
    invalid_roi.convertTo(invalid_roi, CV_8U);

    cv::Mat exterior = invalid_roi.clone();
    cv::floodFill(exterior, cv::Point(0, 0), cv::Scalar(0));
    if (exterior.cols > 1) {
        cv::floodFill(exterior, cv::Point(exterior.cols - 1, 0), cv::Scalar(0));
    }
    if (exterior.rows > 1) {
        cv::floodFill(exterior, cv::Point(0, exterior.rows - 1), cv::Scalar(0));
    }
    if (exterior.cols > 1 && exterior.rows > 1) {
        cv::floodFill(exterior, cv::Point(exterior.cols - 1, exterior.rows - 1), cv::Scalar(0));
    }
    for (int x = 0; x < exterior.cols; ++x) {
        if (exterior.at<uint8_t>(0, x)) {
            cv::floodFill(exterior, cv::Point(x, 0), cv::Scalar(0));
        }
        if (exterior.rows > 1 && exterior.at<uint8_t>(exterior.rows - 1, x)) {
            cv::floodFill(exterior, cv::Point(x, exterior.rows - 1), cv::Scalar(0));
        }
    }
    for (int y = 0; y < exterior.rows; ++y) {
        if (exterior.at<uint8_t>(y, 0)) {
            cv::floodFill(exterior, cv::Point(0, y), cv::Scalar(0));
        }
        if (exterior.cols > 1 && exterior.at<uint8_t>(y, exterior.cols - 1)) {
            cv::floodFill(exterior, cv::Point(exterior.cols - 1, y), cv::Scalar(0));
        }
    }

    metrics.enclosed_hole_pixels = cv::countNonZero(exterior);
    metrics.enclosed_hole_fraction = metrics.valid_bbox_pixels > 0
        ? static_cast<double>(metrics.enclosed_hole_pixels) / metrics.valid_bbox_pixels
        : 0.0;

    const double capped_area_ratio = std::clamp(metrics.area_ratio, 0.0, 1.0);
    const double hole_penalty = 1.0 - std::clamp(metrics.enclosed_hole_fraction, 0.0, 1.0);
    metrics.mesh_completeness_score =
        metrics.target_coverage_fraction *
        capped_area_ratio *
        metrics.valid_bbox_fill_fraction *
        hole_penalty;
}

static SweepMetrics score_surface(QuadSurface& result,
                                  QuadSurface& target,
                                  const PointIndex& target_index,
                                  double target_area_vx2,
                                  double target_area_cm2,
                                  int target_valid_points,
                                  double voxelsize,
                                  double tolerance,
                                  int max_samples)
{
    SweepMetrics metrics;
    metrics.has_result = true;
    metrics.valid_points = result.countValidPoints();
    metrics.valid_quads = result.countValidQuads();
    metrics.area_vx2 = vc::surface::computeSurfaceAreaVox2(result);
    metrics.area_cm2 = metrics.area_vx2 * voxelsize * voxelsize / 1e8;
    if (target_area_vx2 > 0.0) {
        metrics.area_ratio = metrics.area_vx2 / target_area_vx2;
        metrics.area_abs_error_ratio = std::abs(metrics.area_vx2 - target_area_vx2) / target_area_vx2;
    }

    const auto* result_points = result.rawPointsPtr();
    const auto* target_points = target.rawPointsPtr();
    metrics.result_to_target = sampled_distance_stats(*result_points, target_index, metrics.valid_points, max_samples, tolerance);

    PointIndex result_index;
    result_index.buildFromMat(*result_points);
    metrics.target_to_result = sampled_distance_stats(*target_points, result_index, target_valid_points, max_samples, tolerance);

    metrics.result_coverage_fraction = metrics.result_to_target.samples > 0
        ? static_cast<double>(metrics.result_to_target.within_tolerance) / metrics.result_to_target.samples
        : 0.0;
    metrics.target_coverage_fraction = metrics.target_to_result.samples > 0
        ? static_cast<double>(metrics.target_to_result.within_tolerance) / metrics.target_to_result.samples
        : 0.0;
    metrics.coverage_adjusted_mean_distance_vx =
        metrics.result_to_target.mean / std::max(0.05, metrics.target_coverage_fraction);
    add_completeness_metrics(result, metrics);

    const double normalized_distance = tolerance > 0.0 ? metrics.result_to_target.mean / tolerance : metrics.result_to_target.mean;
    metrics.objective_score =
        metrics.area_abs_error_ratio +
        normalized_distance +
        (1.0 - metrics.target_coverage_fraction) +
        0.5 * (1.0 - metrics.result_coverage_fraction);

    (void)target_area_cm2;
    return metrics;
}

static SweepMetrics score_surface_without_target(QuadSurface& result, double voxelsize)
{
    SweepMetrics metrics;
    metrics.has_result = true;
    metrics.valid_points = result.countValidPoints();
    metrics.valid_quads = result.countValidQuads();
    metrics.area_vx2 = vc::surface::computeSurfaceAreaVox2(result);
    metrics.area_cm2 = metrics.area_vx2 * voxelsize * voxelsize / 1e8;
    add_completeness_metrics(result, metrics);
    const double hole_penalty = 1.0 - std::clamp(metrics.enclosed_hole_fraction, 0.0, 1.0);
    metrics.mesh_completeness_score =
        metrics.area_cm2 *
        metrics.valid_bbox_fill_fraction *
        hole_penalty;
    metrics.objective_score = -metrics.mesh_completeness_score;
    return metrics;
}

static nlohmann::json metrics_json(const SweepMetrics& metrics)
{
    return {
        {"has_result", metrics.has_result},
        {"valid_points", metrics.valid_points},
        {"valid_quads", metrics.valid_quads},
        {"area_vx2", metrics.area_vx2},
        {"area_cm2", metrics.area_cm2},
        {"area_ratio", metrics.area_ratio},
        {"area_abs_error_ratio", metrics.area_abs_error_ratio},
        {"result_to_target", distance_stats_json(metrics.result_to_target)},
        {"target_to_result", distance_stats_json(metrics.target_to_result)},
        {"result_coverage_fraction", metrics.result_coverage_fraction},
        {"target_coverage_fraction", metrics.target_coverage_fraction},
        {"coverage_adjusted_mean_distance_vx", metrics.coverage_adjusted_mean_distance_vx},
        {"valid_bbox_pixels", metrics.valid_bbox_pixels},
        {"enclosed_hole_pixels", metrics.enclosed_hole_pixels},
        {"valid_bbox_fill_fraction", metrics.valid_bbox_fill_fraction},
        {"enclosed_hole_fraction", metrics.enclosed_hole_fraction},
        {"mesh_completeness_score", metrics.mesh_completeness_score},
        {"objective_score", metrics.objective_score}
    };
}

static void add_metric_meta(QuadSurface& surf, const SweepMetrics& metrics)
{
    surf.meta["sweep_objective_score"] = metrics.objective_score;
    surf.meta["sweep_area_ratio"] = metrics.area_ratio;
    surf.meta["sweep_target_coverage_fraction"] = metrics.target_coverage_fraction;
    surf.meta["sweep_result_to_target_mean_vx"] = metrics.result_to_target.mean;
    surf.meta["sweep_result_to_target_p95_vx"] = metrics.result_to_target.p95;
    surf.meta["sweep_mesh_completeness_score"] = metrics.mesh_completeness_score;
    surf.meta["sweep_valid_bbox_fill_fraction"] = metrics.valid_bbox_fill_fraction;
    surf.meta["sweep_enclosed_hole_fraction"] = metrics.enclosed_hole_fraction;
}

static std::vector<QuadSurface*> load_source_surfaces(const std::filesystem::path& src_dir,
                                                      QuadSurface* src)
{
    std::vector<QuadSurface*> surfaces;
    for (const auto& entry : std::filesystem::directory_iterator(src_dir)) {
        if (!std::filesystem::is_directory(entry)) {
            continue;
        }

        std::string name = entry.path().filename();
        if (name.compare(0, std::strlen(kAutoGrowPrefix), kAutoGrowPrefix)) {
            continue;
        }

        std::filesystem::path meta_fn = entry.path() / "meta.json";
        if (!std::filesystem::exists(meta_fn)) {
            continue;
        }

        utils::Json meta = utils::Json::parse_file(meta_fn);

        if (!meta.count("bbox")) {
            continue;
        }

        if (meta.value("format", std::string{"NONE"}) != "tifxyz") {
            continue;
        }

        QuadSurface* sm;
        if (src && entry.path().filename() == src->id) {
            sm = src;
        } else {
            sm = new QuadSurface(entry.path(), meta);
        }

        surfaces.push_back(sm);
    }
    return surfaces;
}

static void delete_loaded_source_surfaces(std::vector<QuadSurface*>& surfaces, QuadSurface* src)
{
    for (auto* sm : surfaces) {
        if (sm != src) {
            delete sm;
        }
    }
    surfaces.clear();
}

} // namespace

static void add_target_context(utils::Json& meta, const std::filesystem::path& volume_path)
{
    std::filesystem::path normalized_volume_path = volume_path.lexically_normal();
    if (normalized_volume_path.filename().empty()) {
        normalized_volume_path = normalized_volume_path.parent_path();
    }

    const std::string volume_name = normalized_volume_path.filename().string();
    if (!volume_name.empty() && !meta.contains("target_volume")) {
        meta["target_volume"] = volume_name;
    }

    const std::filesystem::path volumes_dir = normalized_volume_path.parent_path();
    if (volumes_dir.filename() != "volumes") {
        return;
    }

    const std::filesystem::path volpkg_root = volumes_dir.parent_path();
    std::string scroll_name = volpkg_root.filename().string();

    const std::filesystem::path config_path = volpkg_root / "config.json";
    std::error_code ec;
    if (std::filesystem::is_regular_file(config_path, ec)) {
        try {
            auto cfg = utils::Json::parse_file(config_path);
            if (cfg.contains("name") && cfg["name"].is_string()) {
                scroll_name = cfg["name"].get_string();
            }
        } catch (...) {
            // Keep folder-based fallback if config.json cannot be parsed.
        }
    }

    if (!scroll_name.empty() && !meta.contains("scroll_source")) {
        meta["scroll_source"] = scroll_name;
    }
}

static void add_internal_volume_shape(utils::Json& params, const std::vector<size_t>& shape)
{
    if (shape.size() != 3) {
        throw std::runtime_error("Expected zarr dataset shape [z, y, x]");
    }

    auto volume_shape = utils::Json::array();
    volume_shape.push_back(static_cast<int>(shape[0]));
    volume_shape.push_back(static_cast<int>(shape[1]));
    volume_shape.push_back(static_cast<int>(shape[2]));
    params["_volume_shape"] = std::move(volume_shape);
}




int main(int argc, char *argv[])
{
    std::filesystem::path vol_path, src_dir, tgt_dir, params_path, src_path, sweep_target_path, sweep_ranges_path, seed_coords_path;
    std::string sweep_strategy = "grid";
    std::vector<std::string> seed_coord_values;
    double sweep_cutout_cm = 5.0;
    double sweep_metric_tolerance = 10.0;
    double sweep_min_area_cm2 = 1.0;
    int sweep_max_distance_samples = 20000;
    int sweep_max_runs = 0;
    int sweep_max_gen = -1;
    int sweep_max_width = -1;
    int sweep_max_height = -1;
    int cli_max_width = -1;
    uint32_t sweep_seed = 1;

    bool use_old_args = argc == 6 && argv[1][0] != '-' && argv[2][0] != '-' &&
        argv[3][0] != '-' && argv[4][0] != '-' && argv[5][0] != '-';

    if (use_old_args) {
        vol_path = argv[1];
        src_dir = argv[2];
        tgt_dir = argv[3];
        params_path = argv[4];
        src_path = argv[5];
    } else {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
            ("volume,v", po::value<std::string>()->required(), "OME-Zarr volume path")
            ("src-dir,s", po::value<std::string>()->required(), "Directory containing source segments")
            ("target-dir,t", po::value<std::string>()->required(), "Target directory for output traces")
            ("params,p", po::value<std::string>(), "JSON parameters file (required unless --sweep is used)")
            ("src-segment", po::value<std::string>(), "Source segment path to grow from")
            ("seed-coord", po::value<std::vector<std::string>>()->composing(), "3D seed coordinate x,y,z. May be repeated")
            ("seed-coords", po::value<std::string>(), "File containing seed coordinates as JSON [[x,y,z],...] or text lines x,y,z")
            ("max-width", po::value<int>(), "Override grow params max_width")
            ("sweep", po::value<std::string>(), "Target tifxyz surface to fill; enables parameter sweep mode")
            ("sweep-ranges", po::value<std::string>(), "JSON file describing parameter values/ranges for --sweep")
            ("sweep-strategy", po::value<std::string>()->default_value("grid"), "Sweep strategy: grid or lhs")
            ("sweep-max-runs", po::value<int>()->default_value(0), "Maximum runs for --sweep-strategy lhs")
            ("sweep-max-gen", po::value<int>()->default_value(-1), "Maximum grow generations per sweep run")
            ("sweep-max-width", po::value<int>()->default_value(-1), "Seed-only sweep max_width; skips target comparison metrics")
            ("sweep-max-height", po::value<int>()->default_value(-1), "Seed-only sweep max_height")
            ("sweep-min-area", po::value<double>()->default_value(1.0), "Minimum area in cm^2 for seed-only best complete mesh selection")
            ("sweep-seed", po::value<uint32_t>()->default_value(1), "Deterministic seed for sampled sweep strategies")
            ("sweep-cutout-cm", po::value<double>()->default_value(5.0), "Center cutout edge length used as sweep seed")
            ("sweep-metric-tolerance", po::value<double>()->default_value(10.0), "Distance tolerance in voxels for sweep coverage metrics")
            ("sweep-max-distance-samples", po::value<int>()->default_value(20000), "Maximum sampled points per direction for sweep distance metrics");

        po::variables_map vm;
        try {
            po::store(po::parse_command_line(argc, argv, desc), vm);

            if (vm.count("help")) {
                std::cout << desc << std::endl;
                return EXIT_SUCCESS;
            }

            po::notify(vm);
        } catch (const po::error& e) {
            std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
            std::cerr << "usage: " << argv[0]
                      << " --volume <zarr-volume>"
                      << " --src-dir <src-dir>"
                      << " --target-dir <tgt-dir>"
                      << " [--params <json-params>]"
                      << " [--src-segment <src-segment> | --seed-coord x,y,z [--seed-coord x,y,z ...] | --seed-coords <coords> | --sweep <target-tifxyz> --sweep-ranges <ranges-json>]" << std::endl << std::endl;
            std::cerr << desc << std::endl;
            return EXIT_FAILURE;
        }

        vol_path = vm["volume"].as<std::string>();
        src_dir = vm["src-dir"].as<std::string>();
        tgt_dir = vm["target-dir"].as<std::string>();
        if (vm.count("params")) {
            params_path = vm["params"].as<std::string>();
        }
        if (vm.count("src-segment")) {
            src_path = vm["src-segment"].as<std::string>();
        }
        if (vm.count("seed-coord")) {
            seed_coord_values = vm["seed-coord"].as<std::vector<std::string>>();
        }
        if (vm.count("seed-coords")) {
            seed_coords_path = vm["seed-coords"].as<std::string>();
        }
        if (vm.count("max-width")) {
            cli_max_width = vm["max-width"].as<int>();
        }
        if (vm.count("sweep")) {
            sweep_target_path = vm["sweep"].as<std::string>();
        }
        if (vm.count("sweep-ranges")) {
            sweep_ranges_path = vm["sweep-ranges"].as<std::string>();
        }
        sweep_strategy = vm["sweep-strategy"].as<std::string>();
        sweep_max_runs = vm["sweep-max-runs"].as<int>();
        sweep_max_gen = vm["sweep-max-gen"].as<int>();
        sweep_max_width = vm["sweep-max-width"].as<int>();
        sweep_max_height = vm["sweep-max-height"].as<int>();
        sweep_min_area_cm2 = vm["sweep-min-area"].as<double>();
        sweep_seed = vm["sweep-seed"].as<uint32_t>();
        sweep_cutout_cm = vm["sweep-cutout-cm"].as<double>();
        sweep_metric_tolerance = vm["sweep-metric-tolerance"].as<double>();
        sweep_max_distance_samples = vm["sweep-max-distance-samples"].as<int>();
        const bool has_seed_coordinates = !seed_coord_values.empty() || !seed_coords_path.empty();
        if (sweep_target_path.empty() && src_path.empty() && !has_seed_coordinates) {
            std::cerr << "ERROR: --src-segment, --seed-coord/--seed-coords, or --sweep is required" << std::endl;
            return EXIT_FAILURE;
        }
        if (sweep_target_path.empty() && params_path.empty()) {
            std::cerr << "ERROR: --params is required unless --sweep is used" << std::endl;
            return EXIT_FAILURE;
        }
        if (!sweep_target_path.empty() && has_seed_coordinates) {
            std::cerr << "ERROR: --seed-coord/--seed-coords cannot be combined with --sweep" << std::endl;
            return EXIT_FAILURE;
        }
        if (cli_max_width == 0 || cli_max_width < -1) {
            std::cerr << "ERROR: --max-width must be positive when set" << std::endl;
            return EXIT_FAILURE;
        }
        if (!sweep_target_path.empty() && sweep_ranges_path.empty()) {
            std::cerr << "ERROR: --sweep requires --sweep-ranges" << std::endl;
            return EXIT_FAILURE;
        }
        if (sweep_strategy != "grid" && sweep_strategy != "lhs") {
            std::cerr << "ERROR: --sweep-strategy must be 'grid' or 'lhs'" << std::endl;
            return EXIT_FAILURE;
        }
        if (sweep_strategy == "lhs" && sweep_max_runs <= 0) {
            std::cerr << "ERROR: --sweep-strategy lhs requires --sweep-max-runs > 0" << std::endl;
            return EXIT_FAILURE;
        }
        if (sweep_max_gen == 0 || sweep_max_gen < -1) {
            std::cerr << "ERROR: --sweep-max-gen must be positive when set" << std::endl;
            return EXIT_FAILURE;
        }
        if (sweep_max_width == 0 || sweep_max_width < -1) {
            std::cerr << "ERROR: --sweep-max-width must be positive when set" << std::endl;
            return EXIT_FAILURE;
        }
        if (sweep_max_height == 0 || sweep_max_height < -1) {
            std::cerr << "ERROR: --sweep-max-height must be positive when set" << std::endl;
            return EXIT_FAILURE;
        }
        if (!std::isfinite(sweep_min_area_cm2) || sweep_min_area_cm2 < 0.0) {
            std::cerr << "ERROR: --sweep-min-area must be a non-negative cm^2 value" << std::endl;
            return EXIT_FAILURE;
        }
    }

    while (!src_path.empty() && src_path.filename().empty())
        src_path = src_path.parent_path();

    std::vector<SeedCoordinate> seed_coordinates;
    try {
        seed_coordinates.reserve(seed_coord_values.size());
        for (const auto& value : seed_coord_values) {
            seed_coordinates.push_back(parse_seed_coordinate_string(value));
        }
        if (!seed_coords_path.empty()) {
            auto file_coords = parse_seed_coordinates_file(seed_coords_path);
            seed_coordinates.insert(seed_coordinates.end(), file_coords.begin(), file_coords.end());
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing seed coordinates: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    Json params = params_path.empty() ? Json::object() : Json::parse_file(params_path);
    if (cli_max_width > 0) {
        params["max_width"] = cli_max_width;
    }
    // Honor optional CUDA toggle from params (default true)
    if (params.contains("use_cuda")) {
        set_space_tracing_use_cuda(params.value("use_cuda", true));
    } else {
        set_space_tracing_use_cuda(true);
    }
    params["tgt_dir"] = tgt_dir;

    std::unique_ptr<vc::VcDataset> ds = std::make_unique<vc::VcDataset>(vol_path / "0");

    std::cout << "zarr dataset size for scale group 0 " << ds->shape() << std::endl;
    std::cout << "chunk shape shape " << ds->defaultChunkShape() << std::endl;
    add_internal_volume_shape(params, ds->shape());

    float voxelsize = Json::parse_file(vol_path/"meta.json")["voxelsize"].get_float();

    std::filesystem::create_directories(tgt_dir);

    if (!sweep_target_path.empty()) {
        const bool seed_only_sweep = sweep_max_width > 0 || sweep_max_height > 0;
        const int seed_only_max_width = sweep_max_width;
        const int seed_only_max_height = sweep_max_height;
        auto target = load_quad_from_tifxyz(sweep_target_path.string());
        if (!target) {
            std::cerr << "Error: failed to load sweep target tifxyz " << sweep_target_path << std::endl;
            return EXIT_FAILURE;
        }

        auto seed = seed_only_sweep
            ? std::make_unique<QuadSurface>(sweep_target_path)
            : load_or_create_center_cutout(*target, sweep_target_path, sweep_cutout_cm, voxelsize);
        std::vector<QuadSurface*> surfaces = load_source_surfaces(src_dir, seed.get());
        surfaces.push_back(seed.get());

        const int target_valid_points = target->countValidPoints();
        const int target_valid_quads = target->countValidQuads();
        const cv::Size target_grid_size = target->rawPointsPtr()->size();
        const double target_area_vx2 = vc::surface::computeSurfaceAreaVox2(*target);
        const double target_area_cm2 = target_area_vx2 * voxelsize * voxelsize / 1e8;
        PointIndex target_index;
        if (!seed_only_sweep) {
            target_index.buildFromMat(*target->rawPointsPtr());
        }

        std::vector<SweepRun> runs;
        try {
            const nlohmann::json base_params = json_from_utils(params);
            if (sweep_strategy == "lhs") {
                runs = sample_sweep_runs_latin_hypercube(
                    base_params,
                    sweep_ranges_path,
                    sweep_max_runs,
                    sweep_seed);
            } else {
                runs = expand_sweep_runs(base_params, sweep_ranges_path);
            }

            nlohmann::json default_values = nlohmann::json::object();
            default_values["_sweep_default"] = true;
            runs.insert(runs.begin(), SweepRun{base_params, std::move(default_values)});
            if (sweep_strategy == "lhs" && sweep_max_runs > 0 && static_cast<int>(runs.size()) > sweep_max_runs) {
                runs.resize(static_cast<size_t>(sweep_max_runs));
            }
        } catch (const std::exception& e) {
            delete_loaded_source_surfaces(surfaces, seed.get());
            std::cerr << "Error parsing sweep ranges: " << e.what() << std::endl;
            return EXIT_FAILURE;
        }

        std::cout << "Running " << runs.size() << " sweep configurations "
                  << (seed_only_sweep ? "from seed " : "against ")
                  << sweep_target_path << std::endl;
        if (seed_only_sweep) {
            std::cout << "Seed-only sweep bounds";
            if (seed_only_max_width > 0) {
                std::cout << " max_width=" << seed_only_max_width;
            }
            if (seed_only_max_height > 0) {
                std::cout << " max_height=" << seed_only_max_height;
            }
            std::cout << std::endl;
        } else {
            std::cout << "Target valid_points=" << target_valid_points
                      << " valid_quads=" << target_valid_quads
                      << " area_cm2=" << target_area_cm2 << std::endl;
        }

        nlohmann::json report;
        report["target"] = sweep_target_path.string();
        report["seed_only_sweep"] = seed_only_sweep;
        report["strategy"] = sweep_strategy;
        report["max_runs"] = sweep_max_runs;
        report["seed"] = sweep_seed;
        report["target_metrics"] = {
            {"valid_points", target_valid_points},
            {"valid_quads", target_valid_quads},
            {"width_px", target_grid_size.width},
            {"height_px", target_grid_size.height},
            {"area_vx2", target_area_vx2},
            {"area_cm2", target_area_cm2}
        };
        report["cutout"] = {
            {"edge_cm", sweep_cutout_cm},
            {"seed_id", seed->id}
        };
        if (seed_only_sweep) {
            report["seed_only_bounds"] = {
                {"max_width", seed_only_max_width},
                {"max_height", seed_only_max_height}
            };
        }
        report["runs"] = nlohmann::json::array();

        size_t best_idx = 0;
        double best_score = std::numeric_limits<double>::infinity();
        double best_coverage_adjusted_distance_vx = std::numeric_limits<double>::infinity();
        std::string best_uuid;
        size_t best_complete_idx = 0;
        double best_complete_score = -std::numeric_limits<double>::infinity();
        std::string best_complete_uuid;
        const double min_complete_area_cm2 = seed_only_sweep ? sweep_min_area_cm2 : 0.0;

        for (size_t i = 0; i < runs.size(); ++i) {
            std::string uuid = "sweep_" + zero_padded_index(i) + "_" + get_surface_time_str();
            std::filesystem::path seg_dir = tgt_dir / uuid;
            nlohmann::json run_params = runs[i].params;
            run_params["tgt_dir"] = tgt_dir.string();
            if (sweep_max_gen > 0 && !run_params.contains("steps") && !run_params.contains("grow_steps")) {
                run_params["steps"] = sweep_max_gen;
            }
            if (seed_only_sweep) {
                if (seed_only_max_width > 0) {
                    run_params["max_width"] = seed_only_max_width;
                }
                if (seed_only_max_height > 0) {
                    run_params["max_height"] = seed_only_max_height;
                }
            } else {
                const double run_src_step = std::max(1.0, json_number_or(run_params, "src_step", 20.0));
                run_params["max_width"] = static_cast<int>(std::ceil(target_grid_size.width * run_src_step));
                run_params["max_height"] = static_cast<int>(std::ceil(target_grid_size.height * run_src_step));
            }
            if (!seed_only_sweep && std::isfinite(best_coverage_adjusted_distance_vx)) {
                run_params["sweep_prune_distance_enabled"] = true;
                run_params["sweep_prune_target_path"] = sweep_target_path.string();
                run_params["sweep_prune_best_coverage_adjusted_distance_vx"] = best_coverage_adjusted_distance_vx;
                run_params["sweep_prune_target_area_vx2"] = target_area_vx2;
                run_params["sweep_prune_distance_margin"] = 1.5;
                run_params["sweep_prune_search_radius_vx"] = 1000.0;
                run_params["sweep_prune_min_generations"] = 1000;
                run_params["sweep_prune_interval_generations"] = 10;
                run_params["sweep_prune_min_valid_cells"] = 100;
                run_params["sweep_prune_max_samples"] = 2000;
            }

            std::cout << "Sweep " << (i + 1) << "/" << runs.size()
                      << " values=" << runs[i].values.dump() << std::endl;

            QuadSurface* surf = nullptr;
            nlohmann::json run_report;
            run_report["index"] = i;
            run_report["uuid"] = uuid;
            run_report["path"] = seg_dir.string();
            run_report["values"] = runs[i].values;

            try {
                utils::Json run_utils_params = json_to_utils(run_params);
                surf = grow_surf_from_surfs(seed.get(), surfaces, run_utils_params, voxelsize);
                if (surf) {
                    SweepMetrics sweep_metrics = seed_only_sweep
                        ? score_surface_without_target(*surf, voxelsize)
                        : score_surface(
                            *surf,
                            *target,
                            target_index,
                            target_area_vx2,
                            target_area_cm2,
                            target_valid_points,
                            voxelsize,
                            sweep_metric_tolerance,
                            sweep_max_distance_samples);

                    surf->meta["source"] = "vc_grow_seg_from_segments_sweep";
                    surf->meta["vc_grow_seg_from_segments_params"] = json_to_utils(run_params);
                    surf->meta["sweep_values"] = json_to_utils(runs[i].values);
                    surf->meta["sweep_target"] = sweep_target_path.string();
                    surf->meta["sweep_seed_only"] = seed_only_sweep;
                    add_metric_meta(*surf, sweep_metrics);
                    add_target_context(surf->meta, vol_path);
                    surf->save(seg_dir, uuid);

                    run_report["metrics"] = metrics_json(sweep_metrics);
                    if (sweep_metrics.objective_score < best_score) {
                        best_score = sweep_metrics.objective_score;
                        best_idx = i;
                        best_uuid = uuid;
                    }
                    if (!seed_only_sweep &&
                        sweep_metrics.result_to_target.samples > 0 &&
                        std::isfinite(sweep_metrics.coverage_adjusted_mean_distance_vx) &&
                        sweep_metrics.coverage_adjusted_mean_distance_vx < best_coverage_adjusted_distance_vx) {
                        best_coverage_adjusted_distance_vx = sweep_metrics.coverage_adjusted_mean_distance_vx;
                    }
                    if (seed_only_sweep &&
                        sweep_metrics.area_cm2 >= min_complete_area_cm2 &&
                        std::isfinite(sweep_metrics.mesh_completeness_score) &&
                        sweep_metrics.mesh_completeness_score > best_complete_score) {
                        best_complete_score = sweep_metrics.mesh_completeness_score;
                        best_complete_idx = i;
                        best_complete_uuid = uuid;
                    }

                    std::cout << "  score=" << sweep_metrics.objective_score
                              << " area_ratio=" << sweep_metrics.area_ratio
                              << " target_coverage=" << sweep_metrics.target_coverage_fraction
                              << " mean_dist_vx=" << sweep_metrics.result_to_target.mean
                              << " coverage_adjusted_dist_vx=" << sweep_metrics.coverage_adjusted_mean_distance_vx
                              << " mesh_completeness=" << sweep_metrics.mesh_completeness_score
                              << " output=" << seg_dir << std::endl;
                } else {
                    run_report["metrics"] = metrics_json(SweepMetrics{});
                    run_report["error"] = "grow_surf_from_surfs returned null";
                    std::cout << "  no result" << std::endl;
                }
            } catch (const std::exception& e) {
                run_report["error"] = e.what();
                std::cerr << "  error: " << e.what() << std::endl;
            }

            delete surf;
            report["runs"].push_back(std::move(run_report));
        }

        report["best"] = {
            {"index", best_idx},
            {"uuid", best_uuid},
            {"path", best_uuid.empty() ? std::string{} : (tgt_dir / best_uuid).string()},
            {"objective_score", best_score},
            {"best_coverage_adjusted_distance_vx", best_coverage_adjusted_distance_vx}
        };
        if (seed_only_sweep) {
            report["best_complete_mesh"] = {
                {"index", best_complete_idx},
                {"uuid", best_complete_uuid},
                {"path", best_complete_uuid.empty() ? std::string{} : (tgt_dir / best_complete_uuid).string()},
                {"mesh_completeness_score", best_complete_score},
                {"min_area_cm2", min_complete_area_cm2}
            };
        }

        std::ofstream out(tgt_dir / "sweep_results.json");
        out << report.dump(4) << std::endl;

        delete_loaded_source_surfaces(surfaces, seed.get());

        if (!best_uuid.empty()) {
            std::cout << "Best sweep: index=" << best_idx
                      << " uuid=" << best_uuid
                      << " score=" << best_score
                      << " path=" << (tgt_dir / best_uuid) << std::endl;
        } else if (seed_only_sweep && !best_complete_uuid.empty()) {
            std::cout << "Best complete mesh: index=" << best_complete_idx
                      << " uuid=" << best_complete_uuid
                      << " mesh_completeness_score=" << best_complete_score
                      << " path=" << (tgt_dir / best_complete_uuid) << std::endl;
        } else {
            std::cout << "Sweep completed without a successful result" << std::endl;
        }

        return EXIT_SUCCESS;
    }

    if (!seed_coordinates.empty()) {
        std::vector<QuadSurface*> surfaces = load_source_surfaces(src_dir, nullptr);
        if (surfaces.empty()) {
            std::cerr << "Error: no source surfaces found in " << src_dir << std::endl;
            return EXIT_FAILURE;
        }

        bool had_failure = false;
        std::cout << "Running " << seed_coordinates.size() << " coordinate-seeded grow jobs" << std::endl;
        for (const SeedCoordinate& seed : seed_coordinates) {
            QuadSurface* src = find_source_surface_containing_point(surfaces, seed.point);
            if (!src) {
                std::cerr << "Error: no source patch contains seed coordinate "
                          << seed.point << std::endl;
                had_failure = true;
                continue;
            }

            std::cout << "Seed " << seed.label
                      << " point=" << seed.point
                      << " source=" << src->path << std::endl;

            QuadSurface* surf = nullptr;
            try {
                surf = grow_surf_from_surfs(src, surfaces, params, voxelsize);
            } catch (const std::exception& e) {
                std::cerr << "Error growing from seed " << seed.label << ": " << e.what() << std::endl;
                had_failure = true;
                continue;
            }

            if (!surf) {
                std::cerr << "Error: grow_surf_from_surfs returned null for seed " << seed.label << std::endl;
                had_failure = true;
                continue;
            }

            surf->meta["source"] = "vc_grow_seg_from_segments_seed_coordinate";
            surf->meta["vc_grow_seg_from_segments_params"] = utils::Json::parse(params.dump());
            surf->meta["seed_coordinate"] = {
                seed.point[0],
                seed.point[1],
                seed.point[2]
            };
            surf->meta["seed_source_patch"] = src->path.string();
            add_target_context(surf->meta, vol_path);

            std::string uuid = "auto_trace_" + get_surface_time_str() + "_" + seed.label;
            std::filesystem::path seg_dir = tgt_dir / uuid;
            surf->save(seg_dir, uuid);
            std::cout << "  output=" << seg_dir << std::endl;

            delete surf;
        }

        delete_loaded_source_surfaces(surfaces, nullptr);
        return had_failure ? EXIT_FAILURE : EXIT_SUCCESS;
    }

    std::vector<QuadSurface*> surfaces;
    std::filesystem::path meta_fn = src_path / "meta.json";
    if (!std::filesystem::exists(meta_fn)) {
        std::cerr << "Error: meta.json not found at " << meta_fn << std::endl;
        return EXIT_FAILURE;
    }

    utils::Json meta = utils::Json::parse_file(meta_fn);
    QuadSurface *src = new QuadSurface(src_path, meta);

    surfaces = load_source_surfaces(src_dir, src);

    QuadSurface *surf = grow_surf_from_surfs(src, surfaces, params, voxelsize);

    if (!surf)
        return EXIT_SUCCESS;

    surf->meta["source"] = "vc_grow_seg_from_segments";
    surf->meta["vc_grow_seg_from_segments_params"] = utils::Json::parse(params.dump());
    add_target_context(surf->meta, vol_path);
    std::string uuid = "auto_trace_" + get_surface_time_str();;
    std::filesystem::path seg_dir = tgt_dir / uuid;
    surf->save(seg_dir, uuid);

    delete surf;
    delete_loaded_source_surfaces(surfaces, src);
    delete src;

    return EXIT_SUCCESS;
}
