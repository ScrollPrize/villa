#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <omp.h>
#include <opencv2/core.hpp>

namespace fs = std::filesystem;

namespace {

const cv::Vec3f kInvalid(-1.0f, -1.0f, -1.0f);

struct Config {
    fs::path cutDir = "/home/sean/Documents/volpkgs/s1_ds2.volpkg/paths/cut";
    fs::path output = "/home/sean/Documents/volpkgs/s1_ds2.volpkg/paths/cut/w01_to_w29_normal_rays_point_collections.json";
    int startWinding = 1;
    int endWinding = 29;
    float maxDistance = 12000.0f;
    float minHitDistance = 0.5f;
    float dedupRadius = 2.0f;
    float bboxPadding = 1.0f;
    int indexStride = 1;
    int rayStride = 1;
    int normalRadius = 3;
    std::string directionMode = "row-tangent";
    int threads = 0;
    bool rowConstantZ = false;
    bool help = false;
};

struct SurfaceEntry {
    int winding = -1;
    fs::path path;
    SurfacePatchIndex::SurfacePtr surface;
};

struct RayTriangleIntersection {
    float t = 0.0f;
    cv::Vec3f bary = {0.0f, 0.0f, 0.0f};
};

struct RayHit {
    float t = 0.0f;
    cv::Vec3f point = kInvalid;
    int winding = 0;
};

struct RowZOrigin {
    cv::Vec3f point = kInvalid;
    float rowCoord = 0.0f;
};

bool valid_point(const cv::Vec3f& p)
{
    return p[0] != -1.0f && p[1] != -1.0f && p[2] != -1.0f
        && std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

bool finite_vec(const cv::Vec3f& v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

std::optional<cv::Vec3f> sample_surface(const cv::Mat_<cv::Vec3f>& points,
                                         float rowCoord,
                                         float colCoord)
{
    if (!std::isfinite(rowCoord) || !std::isfinite(colCoord) ||
        rowCoord < 0.0f || colCoord < 0.0f ||
        rowCoord > static_cast<float>(points.rows - 1) ||
        colCoord > static_cast<float>(points.cols - 1)) {
        return std::nullopt;
    }

    const int r0 = std::clamp(static_cast<int>(std::floor(rowCoord)), 0, points.rows - 1);
    const int c0 = std::clamp(static_cast<int>(std::floor(colCoord)), 0, points.cols - 1);
    const int r1 = std::min(r0 + 1, points.rows - 1);
    const int c1 = std::min(c0 + 1, points.cols - 1);
    const cv::Vec3f& p00 = points(r0, c0);
    const cv::Vec3f& p01 = points(r0, c1);
    const cv::Vec3f& p10 = points(r1, c0);
    const cv::Vec3f& p11 = points(r1, c1);
    if (!valid_point(p00) || !valid_point(p01) || !valid_point(p10) || !valid_point(p11)) {
        return std::nullopt;
    }

    const float tr = rowCoord - static_cast<float>(r0);
    const float tc = colCoord - static_cast<float>(c0);
    const cv::Vec3f top = p00 * (1.0f - tc) + p01 * tc;
    const cv::Vec3f bottom = p10 * (1.0f - tc) + p11 * tc;
    return top * (1.0f - tr) + bottom * tr;
}

std::optional<int> winding_from_name(const fs::path& path)
{
    static const std::regex re(R"(^w([0-9]+)(?:_|$))");
    std::smatch match;
    const std::string name = path.filename().string();
    if (!std::regex_search(name, match, re)) {
        return std::nullopt;
    }
    return std::stoi(match[1].str());
}

std::vector<SurfaceEntry> load_surfaces(const Config& cfg)
{
    std::vector<SurfaceEntry> entries;
    for (const auto& entry : fs::directory_iterator(cfg.cutDir)) {
        if (!entry.is_directory()) {
            continue;
        }
        auto winding = winding_from_name(entry.path());
        if (!winding || *winding < cfg.startWinding || *winding > cfg.endWinding) {
            continue;
        }
        if (!fs::is_regular_file(entry.path() / "meta.json")) {
            continue;
        }
        entries.push_back(SurfaceEntry{*winding, entry.path(), std::make_shared<QuadSurface>(entry.path())});
    }

    std::sort(entries.begin(), entries.end(), [](const SurfaceEntry& a, const SurfaceEntry& b) {
        return a.winding < b.winding;
    });
    return entries;
}

cv::Vec3f valid_centroid(const cv::Mat_<cv::Vec3f>& points)
{
    cv::Vec3d sum(0.0, 0.0, 0.0);
    std::uint64_t count = 0;
    for (int r = 0; r < points.rows; ++r) {
        for (int c = 0; c < points.cols; ++c) {
            const cv::Vec3f& p = points(r, c);
            if (!valid_point(p)) {
                continue;
            }
            sum += cv::Vec3d(p[0], p[1], p[2]);
            ++count;
        }
    }
    if (count == 0) {
        return kInvalid;
    }
    const double inv = 1.0 / static_cast<double>(count);
    return cv::Vec3f(static_cast<float>(sum[0] * inv),
                     static_cast<float>(sum[1] * inv),
                     static_cast<float>(sum[2] * inv));
}

std::vector<float> row_median_z(const cv::Mat_<cv::Vec3f>& points)
{
    std::vector<float> rowZ(static_cast<std::size_t>(points.rows), std::numeric_limits<float>::quiet_NaN());
    std::vector<float> values;
    values.reserve(static_cast<std::size_t>(points.cols));
    for (int r = 0; r < points.rows; ++r) {
        values.clear();
        for (int c = 0; c < points.cols; ++c) {
            const cv::Vec3f& p = points(r, c);
            if (valid_point(p)) {
                values.push_back(p[2]);
            }
        }
        if (values.empty()) {
            continue;
        }
        const std::size_t mid = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
        rowZ[static_cast<std::size_t>(r)] = values[mid];
    }
    return rowZ;
}

std::vector<cv::Vec3f> row_centroids(const cv::Mat_<cv::Vec3f>& points)
{
    std::vector<cv::Vec3f> centroids(static_cast<std::size_t>(points.rows), kInvalid);
    for (int r = 0; r < points.rows; ++r) {
        cv::Vec3d sum(0.0, 0.0, 0.0);
        std::uint64_t count = 0;
        for (int c = 0; c < points.cols; ++c) {
            const cv::Vec3f& p = points(r, c);
            if (!valid_point(p)) {
                continue;
            }
            sum += cv::Vec3d(p[0], p[1], p[2]);
            ++count;
        }
        if (count == 0) {
            continue;
        }
        const double inv = 1.0 / static_cast<double>(count);
        centroids[static_cast<std::size_t>(r)] = cv::Vec3f(
            static_cast<float>(sum[0] * inv),
            static_cast<float>(sum[1] * inv),
            static_cast<float>(sum[2] * inv));
    }
    return centroids;
}

std::optional<RowZOrigin> interpolate_column_at_z(const cv::Mat_<cv::Vec3f>& points,
                                                  int rowHint,
                                                  int col,
                                                  float z)
{
    if (col < 0 || col >= points.cols || !std::isfinite(z)) {
        return std::nullopt;
    }

    std::optional<RowZOrigin> best;
    float bestRowDistance = std::numeric_limits<float>::max();
    constexpr float eps = 1e-5f;
    for (int r = 0; r + 1 < points.rows; ++r) {
        const cv::Vec3f& p0 = points(r, col);
        const cv::Vec3f& p1 = points(r + 1, col);
        if (!valid_point(p0) || !valid_point(p1)) {
            continue;
        }

        const float z0 = p0[2];
        const float z1 = p1[2];
        const float lo = std::min(z0, z1) - eps;
        const float hi = std::max(z0, z1) + eps;
        if (z < lo || z > hi) {
            continue;
        }

        float t = 0.0f;
        const float dz = z1 - z0;
        if (std::abs(dz) > eps) {
            t = (z - z0) / dz;
        }
        t = std::clamp(t, 0.0f, 1.0f);
        const float rowCoord = static_cast<float>(r) + t;
        const float rowDistance = std::abs(rowCoord - static_cast<float>(rowHint));
        if (rowDistance >= bestRowDistance) {
            continue;
        }

        cv::Vec3f point = p0 * (1.0f - t) + p1 * t;
        point[2] = z;
        best = RowZOrigin{point, rowCoord};
        bestRowDistance = rowDistance;
    }
    return best;
}

cv::Vec3f averaged_grid_normal(const cv::Mat_<cv::Vec3f>& sourcePoints,
                                float rowCoord,
                                float colCoord,
                                int radius)
{
    cv::Vec3f center = grid_normal(sourcePoints, cv::Vec3f(colCoord, rowCoord, 0.0f));
    const float centerLen = cv::norm(center);
    if (!std::isfinite(centerLen) || centerLen <= 1e-6f) {
        return {NAN, NAN, NAN};
    }
    center *= 1.0f / centerLen;

    cv::Vec3f sum(0.0f, 0.0f, 0.0f);
    int count = 0;
    const int r0 = static_cast<int>(std::floor(rowCoord)) - radius;
    const int r1 = static_cast<int>(std::ceil(rowCoord)) + radius;
    const int c0 = static_cast<int>(std::floor(colCoord)) - radius;
    const int c1 = static_cast<int>(std::ceil(colCoord)) + radius;
    for (int rr = r0; rr <= r1; ++rr) {
        if (rr < 0 || rr >= sourcePoints.rows) {
            continue;
        }
        for (int cc = c0; cc <= c1; ++cc) {
            if (cc < 0 || cc >= sourcePoints.cols) {
                continue;
            }
            cv::Vec3f n = grid_normal(sourcePoints, cv::Vec3f(static_cast<float>(cc),
                                                               static_cast<float>(rr),
                                                               0.0f));
            const float len = cv::norm(n);
            if (!std::isfinite(len) || len <= 1e-6f) {
                continue;
            }
            n *= 1.0f / len;
            if (n.dot(center) < 0.0f) {
                n *= -1.0f;
            }
            sum += n;
            ++count;
        }
    }

    if (count == 0) {
        return center;
    }
    const float sumLen = cv::norm(sum);
    if (!std::isfinite(sumLen) || sumLen <= 1e-6f) {
        return center;
    }
    return sum * (1.0f / sumLen);
}

cv::Vec3f row_tangent_direction(const cv::Mat_<cv::Vec3f>& sourcePoints,
                                float rowCoord,
                                float colCoord,
                                int radius)
{
    cv::Vec3f tangent(0.0f, 0.0f, 0.0f);
    int count = 0;
    const int maxStep = std::max(1, radius);
    for (int step = 1; step <= maxStep; ++step) {
        auto left = sample_surface(sourcePoints, rowCoord, colCoord - static_cast<float>(step));
        auto right = sample_surface(sourcePoints, rowCoord, colCoord + static_cast<float>(step));
        if (!left || !right) {
            continue;
        }
        cv::Vec3f t = *right - *left;
        t[2] = 0.0f;
        const float len = cv::norm(t);
        if (!std::isfinite(len) || len <= 1e-6f) {
            continue;
        }
        t *= 1.0f / len;
        if (count > 0 && t.dot(tangent) < 0.0f) {
            t *= -1.0f;
        }
        tangent += t;
        ++count;
    }

    if (count == 0) {
        const int row = std::clamp(static_cast<int>(std::round(rowCoord)), 0, sourcePoints.rows - 1);
        const int col = std::clamp(static_cast<int>(std::round(colCoord)), 0, sourcePoints.cols - 1);
        std::vector<int> validCols;
        validCols.reserve(static_cast<std::size_t>(sourcePoints.cols));
        for (int cc = 0; cc < sourcePoints.cols; ++cc) {
            if (valid_point(sourcePoints(row, cc))) {
                validCols.push_back(cc);
            }
        }
        if (validCols.size() >= 2) {
            auto upper = std::lower_bound(validCols.begin(), validCols.end(), col);
            int c0 = -1;
            int c1 = -1;
            if (upper == validCols.begin()) {
                c0 = validCols[0];
                c1 = validCols[1];
            } else if (upper == validCols.end()) {
                c0 = validCols[validCols.size() - 2];
                c1 = validCols[validCols.size() - 1];
            } else {
                c0 = *(upper - 1);
                c1 = *upper;
                if (c1 == col && upper + 1 != validCols.end()) {
                    c0 = col;
                    c1 = *(upper + 1);
                }
            }
            if (c0 >= 0 && c1 >= 0 && c0 != c1) {
                tangent = sourcePoints(row, c1) - sourcePoints(row, c0);
                tangent[2] = 0.0f;
                count = 1;
            }
        }
    }
    if (count == 0) {
        return {NAN, NAN, NAN};
    }
    const float tangentLen = cv::norm(tangent);
    if (!std::isfinite(tangentLen) || tangentLen <= 1e-6f) {
        return {NAN, NAN, NAN};
    }
    tangent *= 1.0f / tangentLen;

    cv::Vec3f dir(-tangent[1], tangent[0], 0.0f);
    const float dirLen = cv::norm(dir);
    if (!std::isfinite(dirLen) || dirLen <= 1e-6f) {
        return {NAN, NAN, NAN};
    }
    return dir * (1.0f / dirLen);
}

cv::Vec3f outward_direction_from_row_centroid(const cv::Mat_<cv::Vec3f>& sourcePoints,
                                             const std::vector<cv::Vec3f>& rowCentroids,
                                             const cv::Vec3f& origin,
                                             float rowCoord,
                                             float colCoord,
                                             int rowHint,
                                             int normalRadius,
                                             const std::string& directionMode)
{
    if (!valid_point(origin)) {
        throw std::runtime_error("Cannot orient ray from invalid origin");
    }

    cv::Vec3f dir = directionMode == "normal"
        ? averaged_grid_normal(sourcePoints, rowCoord, colCoord, std::max(0, normalRadius))
        : row_tangent_direction(sourcePoints, rowCoord, colCoord, std::max(1, normalRadius));
    dir[2] = 0.0f;
    const float dirLen = cv::norm(dir);
    if (!std::isfinite(dirLen) || dirLen <= 1e-6f) {
        std::ostringstream msg;
        msg << "Cannot compute " << directionMode << " ray direction at row "
            << rowHint << " col " << colCoord;
        throw std::runtime_error(msg.str());
    }
    dir *= 1.0f / dirLen;

    if (rowHint < 0 || rowHint >= static_cast<int>(rowCentroids.size()) ||
        !valid_point(rowCentroids[static_cast<std::size_t>(rowHint)])) {
        std::ostringstream msg;
        msg << "Cannot compute same-Z centroid for row " << rowHint;
        throw std::runtime_error(msg.str());
    }

    cv::Vec3f outward = origin - rowCentroids[static_cast<std::size_t>(rowHint)];
    outward[2] = 0.0f;
    const float outwardLen = cv::norm(outward);
    if (!std::isfinite(outwardLen) || outwardLen <= 1e-6f) {
        std::ostringstream msg;
        msg << "Cannot orient ray at row " << rowHint << " col " << colCoord
            << ": origin is coincident with same-Z centroid";
        throw std::runtime_error(msg.str());
    }
    outward *= 1.0f / outwardLen;

    if (dir.dot(outward) < 0.0f) {
        dir *= -1.0f;
    }
    return dir;
}
std::optional<RayTriangleIntersection> ray_triangle_intersection(const cv::Vec3f& origin,
                                                                 const cv::Vec3f& dir,
                                                                 const std::array<cv::Vec3f, 3>& tri,
                                                                 float minT,
                                                                 float maxT)
{
    constexpr float eps = 1e-6f;
    const cv::Vec3f e1 = tri[1] - tri[0];
    const cv::Vec3f e2 = tri[2] - tri[0];
    const cv::Vec3f p = dir.cross(e2);
    const float det = e1.dot(p);
    if (std::abs(det) < eps) {
        return std::nullopt;
    }

    const float invDet = 1.0f / det;
    const cv::Vec3f tv = origin - tri[0];
    const float u = tv.dot(p) * invDet;
    if (u < -eps || u > 1.0f + eps) {
        return std::nullopt;
    }

    const cv::Vec3f q = tv.cross(e1);
    const float v = dir.dot(q) * invDet;
    if (v < -eps || u + v > 1.0f + eps) {
        return std::nullopt;
    }

    const float t = e2.dot(q) * invDet;
    if (t < minT || t > maxT || !std::isfinite(t)) {
        return std::nullopt;
    }
    return RayTriangleIntersection{t, {1.0f - u - v, u, v}};
}

void dedup_hits(std::vector<RayHit>& hits, float radius)
{
    std::sort(hits.begin(), hits.end(), [](const RayHit& a, const RayHit& b) {
        return a.t < b.t;
    });
    if (radius <= 0.0f || hits.empty()) {
        return;
    }

    std::vector<RayHit> deduped;
    deduped.reserve(hits.size());
    std::size_t clusterBegin = 0;
    while (clusterBegin < hits.size()) {
        std::size_t clusterEnd = clusterBegin + 1;
        float clusterSum = hits[clusterBegin].t;
        while (clusterEnd < hits.size() &&
               hits[clusterEnd].t - hits[clusterBegin].t <= radius) {
            clusterSum += hits[clusterEnd].t;
            ++clusterEnd;
        }

        const float mean = clusterSum / static_cast<float>(clusterEnd - clusterBegin);
        std::size_t best = clusterBegin;
        float bestErr = std::abs(hits[clusterBegin].t - mean);
        for (std::size_t i = clusterBegin + 1; i < clusterEnd; ++i) {
            const float err = std::abs(hits[i].t - mean);
            if (err < bestErr) {
                best = i;
                bestErr = err;
            }
        }
        deduped.push_back(hits[best]);
        clusterBegin = clusterEnd;
    }
    hits.swap(deduped);
}

std::string json_escape(const std::string& s)
{
    std::ostringstream out;
    for (const char ch : s) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(ch))
                        << std::dec << std::setfill(' ');
                } else {
                    out << ch;
                }
        }
    }
    return out.str();
}

void write_point(std::ostream& out, std::uint64_t pointId, const cv::Vec3f& p, int winding)
{
    out << "        \"" << pointId << "\": {\n"
        << "          \"p\": [" << p[0] << ", " << p[1] << ", " << p[2] << "],\n"
        << "          \"creation_time\": 0,\n"
        << "          \"wind_a\": " << winding << "\n"
        << "        }";
}

Config parse_args(int argc, char* argv[])
{
    Config cfg;
    auto require_value = [&](int& i, const std::string& option) -> std::string {
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value for " + option);
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            cfg.help = true;
            return cfg;
        } else if (arg == "--cut-dir") {
            cfg.cutDir = require_value(i, arg);
        } else if (arg == "--output") {
            cfg.output = require_value(i, arg);
        } else if (arg == "--start-winding") {
            cfg.startWinding = std::stoi(require_value(i, arg));
        } else if (arg == "--end-winding") {
            cfg.endWinding = std::stoi(require_value(i, arg));
        } else if (arg == "--max-distance") {
            cfg.maxDistance = std::stof(require_value(i, arg));
        } else if (arg == "--min-hit-distance") {
            cfg.minHitDistance = std::stof(require_value(i, arg));
        } else if (arg == "--dedup-radius") {
            cfg.dedupRadius = std::stof(require_value(i, arg));
        } else if (arg == "--bbox-padding") {
            cfg.bboxPadding = std::stof(require_value(i, arg));
        } else if (arg == "--index-stride") {
            cfg.indexStride = std::stoi(require_value(i, arg));
        } else if (arg == "--ray-stride") {
            cfg.rayStride = std::stoi(require_value(i, arg));
        } else if (arg == "--normal-radius") {
            cfg.normalRadius = std::stoi(require_value(i, arg));
        } else if (arg == "--direction-mode") {
            cfg.directionMode = require_value(i, arg);
            if (cfg.directionMode != "row-tangent" && cfg.directionMode != "normal") {
                throw std::runtime_error("--direction-mode must be row-tangent or normal");
            }
        } else if (arg == "--row-constant-z") {
            cfg.rowConstantZ = true;
        } else if (arg == "--threads") {
            cfg.threads = std::stoi(require_value(i, arg));
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    return cfg;
}

void print_usage(const char* argv0)
{
    std::cout
        << "usage: " << argv0 << " [options]\n"
        << "  --cut-dir DIR          tifxyz winding directory root\n"
        << "  --output JSON          output VC3D point collection JSON\n"
        << "  --start-winding N      source winding, default 1\n"
        << "  --end-winding N        target winding, default 29\n"
        << "  --max-distance V       ray distance in voxels, default 12000\n"
        << "  --min-hit-distance V   skip ray hits this close to origin, default 0.5\n"
        << "  --dedup-radius V       merge duplicate triangle hits, default 2\n"
        << "  --index-stride N       SurfacePatchIndex stride, default 1\n"
        << "  --ray-stride N         cast from every Nth source row and column, default 1\n"
        << "  --normal-radius N      direction smoothing radius in grid cells, default 3\n"
        << "  --direction-mode MODE  row-tangent or normal, default row-tangent\n"
        << "  --row-constant-z      use one median source Z per row for all ray origins\n"
        << "  --threads N            OpenMP threads, default runtime\n";
}

} // namespace

int main(int argc, char* argv[])
{
    try {
        Config cfg = parse_args(argc, argv);
        if (cfg.help) {
            print_usage(argv[0]);
            return 0;
        }
        if (cfg.threads > 0) {
            omp_set_num_threads(cfg.threads);
        }
        cfg.rayStride = std::max(1, cfg.rayStride);
        cfg.normalRadius = std::max(0, cfg.normalRadius);

        std::vector<SurfaceEntry> entries = load_surfaces(cfg);
        if (entries.empty()) {
            throw std::runtime_error("No winding tifxyz directories found in requested range");
        }
        auto sourceIt = std::find_if(entries.begin(), entries.end(), [&](const SurfaceEntry& e) {
            return e.winding == cfg.startWinding;
        });
        auto targetIt = std::find_if(entries.begin(), entries.end(), [&](const SurfaceEntry& e) {
            return e.winding == cfg.endWinding;
        });
        if (sourceIt == entries.end() || targetIt == entries.end()) {
            throw std::runtime_error("Missing source or target winding in input directory");
        }

        std::vector<SurfacePatchIndex::SurfacePtr> surfaces;
        surfaces.reserve(entries.size());
        std::unordered_map<QuadSurface*, int> windingBySurface;
        windingBySurface.reserve(entries.size());
        for (const SurfaceEntry& entry : entries) {
            surfaces.push_back(entry.surface);
            windingBySurface.emplace(entry.surface.get(), entry.winding);
            std::cout << "Loaded w" << std::setw(2) << std::setfill('0') << entry.winding
                      << std::setfill(' ') << ": " << entry.path << '\n';
        }

        SurfacePatchIndex index;
        index.setSamplingStride(std::max(1, cfg.indexStride));
        index.rebuild(surfaces, cfg.bboxPadding);
        std::cout << "SurfacePatchIndex: surfaces=" << index.surfaceCount()
                  << " patches=" << index.patchCount()
                  << " stride=" << index.samplingStride() << '\n';

        const cv::Mat_<cv::Vec3f> sourcePoints = sourceIt->surface->rawPoints();
        const std::vector<cv::Vec3f> sourceRowCentroids = row_centroids(sourcePoints);
        const std::vector<float> sourceRowZ = cfg.rowConstantZ
            ? row_median_z(sourcePoints)
            : std::vector<float>{};

        std::ofstream out(cfg.output);
        if (!out) {
            throw std::runtime_error("Failed to open output: " + cfg.output.string());
        }
        out << std::setprecision(9);
        out << "{\n";
        out << "  \"vc_pointcollections_json_version\": \"1\",\n";
        out << "  \"collections\": {\n";

        std::uint64_t collectionId = 1;
        std::uint64_t pointId = 1;
        std::uint64_t raysWritten = 0;
        std::uint64_t totalPoints = 0;
        bool firstCollection = true;

        for (int r = 0; r < sourcePoints.rows; r += cfg.rayStride) {
            for (int c = 0; c < sourcePoints.cols; c += cfg.rayStride) {
                cv::Vec3f origin = sourcePoints(r, c);
                float rowCoord = static_cast<float>(r);
                if (!valid_point(origin)) {
                    continue;
                }
                if (cfg.rowConstantZ) {
                    const float z = sourceRowZ[static_cast<std::size_t>(r)];
                    auto interpolated = interpolate_column_at_z(sourcePoints, r, c, z);
                    if (!interpolated) {
                        continue;
                    }
                    origin = interpolated->point;
                    rowCoord = interpolated->rowCoord;
                }
                const cv::Vec3f dir = outward_direction_from_row_centroid(sourcePoints, sourceRowCentroids,
                                                                           origin, rowCoord,
                                                                           static_cast<float>(c), r,
                                                                           cfg.normalRadius,
                                                                           cfg.directionMode);

                std::vector<RayHit> hits;
                const cv::Vec3f end = origin + dir * cfg.maxDistance;
                SurfacePatchIndex::RayQuery query;
                query.src = origin;
                query.end = end;
                query.minT = cfg.minHitDistance;
                query.bboxPadding = cfg.bboxPadding;
                index.forEachTriangle(query, [&](const SurfacePatchIndex::TriangleCandidate& tri) {
                    if (tri.surface == sourceIt->surface) {
                        return;
                    }
                    auto intersection = ray_triangle_intersection(origin, dir, tri.world,
                                                                  cfg.minHitDistance,
                                                                  cfg.maxDistance);
                    if (!intersection) {
                        return;
                    }
                    auto windingIt = windingBySurface.find(tri.surface.get());
                    if (windingIt == windingBySurface.end()) {
                        return;
                    }
                    hits.push_back(RayHit{intersection->t,
                                          origin + dir * intersection->t,
                                          windingIt->second - cfg.startWinding});
                });
                dedup_hits(hits, cfg.dedupRadius);
                if (hits.empty()) {
                    continue;
                }

                if (!firstCollection) {
                    out << ",\n";
                }
                firstCollection = false;

                const std::string name = "w01_normal_ray_r" + std::to_string(r) + "_c" + std::to_string(c);
                out << "    \"" << collectionId << "\": {\n";
                out << "      \"name\": \"" << json_escape(name) << "\",\n";
                out << "      \"points\": {\n";

                write_point(out, pointId++, origin, 0);
                for (const RayHit& hit : hits) {
                    out << ",\n";
                    write_point(out, pointId++, hit.point, hit.winding);
                }

                out << "\n";
                out << "      },\n";
                out << "      \"metadata\": {\"winding_is_absolute\": false},\n";
                out << "      \"color\": [0.1, 0.65, 1.0],\n";
                out << "      \"tags\": {\"source\": \"normal_ray\", \"start_winding\": \""
                    << cfg.startWinding << "\", \"end_winding\": \"" << cfg.endWinding
                    << "\", \"row_constant_z\": \"" << (cfg.rowConstantZ ? "true" : "false")
                    << "\", \"normal_radius\": \"" << cfg.normalRadius
                    << "\", \"direction_mode\": \"" << cfg.directionMode << "\"}\n";
                out << "    }";

                ++collectionId;
                ++raysWritten;
                totalPoints += 1 + hits.size();
            }
        }

        out << "\n";
        out << "  }\n";
        out << "}\n";

        std::cout << "Wrote " << raysWritten << " collections and "
                  << totalPoints << " points to " << cfg.output << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
}
