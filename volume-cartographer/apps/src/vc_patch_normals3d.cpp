#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <omp.h>
#include <opencv2/core.hpp>

#include "utils/Json.hpp"
#include "vc/core/types/VcDataset.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/core/util/Umbilicus.hpp"
#include "vc/core/util/Zarr.hpp"

namespace fs = std::filesystem;
namespace po = boost::program_options;

namespace {

using Json = utils::Json;
using vc::core::util::Umbilicus;

struct Config {
    fs::path patchesDir;
    fs::path referenceZarr;
    fs::path outputZarr;
    fs::path umbilicusPath;
    std::optional<fs::path> metricsJson;
    int level = 0;
    int sampleStep = 16;
    int chunkSize = 64;
    int surfaceStride = 1;
    float radius = 192.0f;
    float minRadius = 32.0f;
    float radiusGrowth = 1.5f;
    int targetSupport = 256;
    float clusterAngleDeg = 25.0f;
    float dominanceRatio = 1.0f;
    int minSupport = 1;
    int previewLimit = 0;
    std::optional<std::array<int, 6>> cropXyz;
};

struct Layout {
    std::vector<size_t> volumeShapeZyx;
    std::vector<size_t> outputShapeZyx;
    std::vector<size_t> chunkShapeZyx;
};

struct Cluster {
    cv::Vec3d sum{0.0, 0.0, 0.0};
    double weight = 0.0;
    int count = 0;
};

struct CellAccumulator {
    std::vector<Cluster> clusters;
    int observations = 0;

    void add(const cv::Vec3d& nIn, double weight, double cosThreshold)
    {
        if (!(weight > 0.0)) {
            return;
        }
        cv::Vec3d n = nIn;
        const double nNorm = std::sqrt(n.dot(n));
        if (!(nNorm > 1e-12)) {
            return;
        }
        n *= 1.0 / nNorm;
        ++observations;

        int bestIdx = -1;
        double bestAbsDot = -1.0;
        for (int i = 0; i < static_cast<int>(clusters.size()); ++i) {
            cv::Vec3d dir = clusters[static_cast<size_t>(i)].sum;
            const double dNorm = std::sqrt(dir.dot(dir));
            if (!(dNorm > 1e-12)) {
                continue;
            }
            dir *= 1.0 / dNorm;
            const double d = n.dot(dir);
            const double ad = std::abs(d);
            if (ad > bestAbsDot) {
                bestAbsDot = ad;
                bestIdx = i;
            }
        }

        if (bestIdx < 0 || bestAbsDot < cosThreshold) {
            Cluster c;
            c.sum = n * weight;
            c.weight = weight;
            c.count = 1;
            clusters.push_back(c);
            return;
        }

        Cluster& c = clusters[static_cast<size_t>(bestIdx)];
        cv::Vec3d dir = c.sum;
        const double dNorm = std::sqrt(dir.dot(dir));
        if (dNorm > 1e-12 && n.dot(dir) < 0.0) {
            n = -n;
        }
        c.sum += n * weight;
        c.weight += weight;
        c.count += 1;
    }

    void mergeCluster(const Cluster& src, double cosThreshold)
    {
        const double srcNorm = std::sqrt(src.sum.dot(src.sum));
        if (!(srcNorm > 1e-12) || !(src.weight > 0.0)) {
            return;
        }
        cv::Vec3d n = src.sum * (1.0 / srcNorm);

        int bestIdx = -1;
        double bestAbsDot = -1.0;
        for (int i = 0; i < static_cast<int>(clusters.size()); ++i) {
            cv::Vec3d dir = clusters[static_cast<size_t>(i)].sum;
            const double dNorm = std::sqrt(dir.dot(dir));
            if (!(dNorm > 1e-12)) {
                continue;
            }
            dir *= 1.0 / dNorm;
            const double ad = std::abs(n.dot(dir));
            if (ad > bestAbsDot) {
                bestAbsDot = ad;
                bestIdx = i;
            }
        }

        observations += src.count;
        if (bestIdx < 0 || bestAbsDot < cosThreshold) {
            clusters.push_back(src);
            return;
        }

        Cluster& dst = clusters[static_cast<size_t>(bestIdx)];
        if (dst.sum.dot(src.sum) < 0.0) {
            dst.sum -= src.sum;
        } else {
            dst.sum += src.sum;
        }
        dst.weight += src.weight;
        dst.count += src.count;
    }

    void mergeFrom(const CellAccumulator& src, double cosThreshold)
    {
        for (const Cluster& c : src.clusters) {
            mergeCluster(c, cosThreshold);
        }
    }
};

struct ChunkResult {
    std::array<size_t, 3> off{};
    std::array<size_t, 3> shape{};
    std::vector<uint8_t> x;
    std::vector<uint8_t> y;
    std::vector<uint8_t> z;
    std::vector<uint8_t> support;
    std::vector<uint8_t> consensus;
    std::vector<uint8_t> angularDispersion;
    uint64_t valid = 0;
    uint64_t emptyCells = 0;
    uint64_t lowSupportCells = 0;
    uint64_t ambiguousCells = 0;
    uint64_t queriedTriangles = 0;
    uint64_t binnedObservations = 0;
    double querySeconds = 0.0;
    double binSeconds = 0.0;
};

static uint8_t encode_dir_component(double v)
{
    if (!std::isfinite(v)) {
        return 128;
    }
    const int q = static_cast<int>(std::lround(std::clamp(v, -1.0, 1.0) * 127.0 + 128.0));
    return static_cast<uint8_t>(std::clamp(q, 0, 255));
}

static uint8_t encode_unit(double v)
{
    if (!std::isfinite(v)) {
        return 0;
    }
    const int q = static_cast<int>(std::lround(std::clamp(v, 0.0, 1.0) * 255.0));
    return static_cast<uint8_t>(std::clamp(q, 0, 255));
}

static uint8_t encode_count(int v)
{
    return static_cast<uint8_t>(std::clamp(v, 0, 255));
}

static std::vector<double> make_radius_buckets(const Config& cfg)
{
    std::vector<double> radii;
    const double minR = std::max(1.0f, std::min(cfg.minRadius, cfg.radius));
    const double maxR = std::max(minR, static_cast<double>(cfg.radius));
    const double growth = std::max(1.01f, cfg.radiusGrowth);
    for (double r = minR; r < maxR; r *= growth) {
        if (radii.empty() || r > radii.back() * 1.001) {
            radii.push_back(r);
        }
        if (radii.size() > 64) {
            break;
        }
    }
    if (radii.empty() || radii.back() < maxR) {
        radii.push_back(maxR);
    }
    return radii;
}

static double point_triangle_distance_sq(
    const cv::Vec3d& p,
    const cv::Vec3d& a,
    const cv::Vec3d& b,
    const cv::Vec3d& c)
{
    const cv::Vec3d ab = b - a;
    const cv::Vec3d ac = c - a;
    const cv::Vec3d ap = p - a;
    const double d1 = ab.dot(ap);
    const double d2 = ac.dot(ap);
    if (d1 <= 0.0 && d2 <= 0.0) return (p - a).dot(p - a);

    const cv::Vec3d bp = p - b;
    const double d3 = ab.dot(bp);
    const double d4 = ac.dot(bp);
    if (d3 >= 0.0 && d4 <= d3) return (p - b).dot(p - b);

    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        const double v = d1 / (d1 - d3);
        const cv::Vec3d proj = a + v * ab;
        return (p - proj).dot(p - proj);
    }

    const cv::Vec3d cp = p - c;
    const double d5 = ab.dot(cp);
    const double d6 = ac.dot(cp);
    if (d6 >= 0.0 && d5 <= d6) return (p - c).dot(p - c);

    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        const double w = d2 / (d2 - d6);
        const cv::Vec3d proj = a + w * ac;
        return (p - proj).dot(p - proj);
    }

    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        const cv::Vec3d proj = b + w * (c - b);
        return (p - proj).dot(p - proj);
    }

    const cv::Vec3d n = ab.cross(ac);
    const double n2 = n.dot(n);
    if (!(n2 > 1e-18)) return std::numeric_limits<double>::infinity();
    const double dist = n.dot(ap);
    return (dist * dist) / n2;
}

static void print_usage()
{
    std::cout
        << "vc_patch_normals3d: Generate normal3d Zarr from tifxyz patches.\n\n"
        << "Usage: vc_patch_normals3d -p <paths_dir> -r <volume.zarr> -u <umbilicus.json> -o <out.zarr> [options]\n\n"
        << "Options:\n"
        << "  -p, --patches PATH       Directory containing tifxyz patch subdirectories\n"
        << "  -r, --reference-zarr PATH Reference OME-Zarr volume root\n"
        << "  -u, --umbilicus PATH     Umbilicus JSON for outward sign orientation\n"
        << "  -o, --output-zarr PATH   Output normal3d Zarr root\n"
        << "      --level N            Reference Zarr level (default: 0)\n"
        << "      --step N             Output sample step in voxels (default: 16)\n"
        << "      --radius R           Max adaptive support radius in voxels (default: 192)\n"
        << "      --min-radius R       Initial adaptive support radius in voxels (default: 32)\n"
        << "      --radius-growth R    Adaptive radius multiplier (default: 1.5)\n"
        << "      --target-support N   Stop radius growth after N observations (default: 256)\n"
        << "      --crop x0 y0 z0 x1 y1 z1  Process only this voxel crop, half-open\n"
        << "      --chunk-size N       Output chunk edge in samples (default: 64)\n"
        << "      --surface-stride N   Surface triangulation stride for SurfacePatchIndex (default: 1)\n"
        << "      --cluster-angle DEG  Sign-invariant cluster angle (default: 25)\n"
        << "      --dominance-ratio R  Best/second cluster ratio required (default: 1.0)\n"
        << "      --min-support N      Minimum winning-cluster observations (default: 1)\n"
        << "      --limit N            Load only first N patches, for smoke tests\n"
        << "      --metrics-json PATH  Write metrics JSON\n";
}

static bool bbox_intersects_crop(const Json& meta, const std::optional<std::array<int, 6>>& cropXyz)
{
    if (!cropXyz.has_value()) {
        return true;
    }
    if (!meta.contains("bbox")) {
        return true;
    }
    const auto& b = meta["bbox"];
    if (!b.is_array() || b.size() < 2) {
        return true;
    }
    const auto& c = *cropXyz;
    const double bx0 = b[0][0].get_double();
    const double by0 = b[0][1].get_double();
    const double bz0 = b[0][2].get_double();
    const double bx1 = b[1][0].get_double();
    const double by1 = b[1][1].get_double();
    const double bz1 = b[1][2].get_double();
    return !(bx1 < c[0] || bx0 > c[3] ||
             by1 < c[1] || by0 > c[4] ||
             bz1 < c[2] || bz0 > c[5]);
}

static std::vector<SurfacePatchIndex::SurfacePtr> discover_surfaces(
    const fs::path& patchesDir,
    int limit,
    const std::optional<std::array<int, 6>>& cropXyz)
{
    std::vector<SurfacePatchIndex::SurfacePtr> surfaces;
    if (!fs::is_directory(patchesDir)) {
        throw std::runtime_error("Patch directory does not exist: " + patchesDir.string());
    }

    std::vector<fs::path> dirs;
    size_t scannedDirs = 0;
    auto lastScanReport = std::chrono::steady_clock::now();
    for (const auto& entry : fs::directory_iterator(patchesDir)) {
        ++scannedDirs;
        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastScanReport).count() >= 5) {
            lastScanReport = now;
            std::cout << "Scanned patch dirs: " << scannedDirs
                      << " candidates_with_files=" << dirs.size() << std::endl;
        }
        if (!entry.is_directory()) continue;
        const fs::path dir = entry.path();
        if (!fs::exists(dir / "meta.json") ||
            !fs::exists(dir / "x.tif") ||
            !fs::exists(dir / "y.tif") ||
            !fs::exists(dir / "z.tif")) {
            continue;
        }
        dirs.push_back(dir);
    }
    std::sort(dirs.begin(), dirs.end());
    std::cout << "Patch dirs with tifxyz files: " << dirs.size() << std::endl;

    size_t parsed = 0;
    size_t skippedOutsideCrop = 0;
    auto lastMetaReport = std::chrono::steady_clock::now();
    for (const auto& dir : dirs) {
        try {
            Json meta = Json::parse_file(dir / "meta.json");
            ++parsed;
            if (meta.value("format", std::string{}) != "tifxyz") {
                continue;
            }
            if (!bbox_intersects_crop(meta, cropXyz)) {
                ++skippedOutsideCrop;
                continue;
            }
            surfaces.push_back(std::make_shared<QuadSurface>(dir, meta));
            if (limit > 0 && static_cast<int>(surfaces.size()) >= limit) {
                break;
            }
            const auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - lastMetaReport).count() >= 5) {
                lastMetaReport = now;
                std::cout << "Parsed patch metadata: " << parsed << "/" << dirs.size()
                          << " selected=" << surfaces.size()
                          << " skipped_outside_crop=" << skippedOutsideCrop << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Skipping patch " << dir << ": " << e.what() << "\n";
        }
    }
    std::cout << "Patch metadata complete: parsed=" << parsed
              << " selected=" << surfaces.size()
              << " skipped_outside_crop=" << skippedOutsideCrop << std::endl;
    return surfaces;
}

static Layout make_layout(const fs::path& referenceZarr, int level, int sampleStep, int chunkSize)
{
    vc::VcDataset ref(referenceZarr / std::to_string(level));
    const auto shape = ref.shape();
    if (shape.size() != 3) {
        throw std::runtime_error("Reference Zarr level is not 3D: " + (referenceZarr / std::to_string(level)).string());
    }
    Layout layout;
    layout.volumeShapeZyx = shape;
    layout.outputShapeZyx = {
        (shape[0] + static_cast<size_t>(sampleStep) - 1) / static_cast<size_t>(sampleStep),
        (shape[1] + static_cast<size_t>(sampleStep) - 1) / static_cast<size_t>(sampleStep),
        (shape[2] + static_cast<size_t>(sampleStep) - 1) / static_cast<size_t>(sampleStep),
    };
    layout.chunkShapeZyx = {
        std::min(static_cast<size_t>(chunkSize), layout.outputShapeZyx[0]),
        std::min(static_cast<size_t>(chunkSize), layout.outputShapeZyx[1]),
        std::min(static_cast<size_t>(chunkSize), layout.outputShapeZyx[2]),
    };
    return layout;
}

static Umbilicus load_umbilicus_for_volume(const fs::path& path, const std::vector<size_t>& volumeShapeZyx)
{
    const cv::Vec3i volumeShape(
        static_cast<int>(volumeShapeZyx[0]),
        static_cast<int>(volumeShapeZyx[1]),
        static_cast<int>(volumeShapeZyx[2]));

    try {
        return Umbilicus::FromFile(path, volumeShape);
    } catch (const std::exception&) {
        Json document = Json::parse_file(path);
        if (!document.is_object() || !document.contains("control_points") || !document["control_points"].is_array()) {
            throw;
        }

        std::vector<cv::Vec3f> points;
        for (const auto& entry : document["control_points"]) {
            if (!entry.is_object() || !entry.contains("x") || !entry.contains("y") || !entry.contains("z")) {
                throw std::runtime_error("Umbilicus control_points entries must contain x/y/z");
            }
            points.emplace_back(
                entry["x"].get_float(),
                entry["y"].get_float(),
                entry["z"].get_float());
        }
        if (points.empty()) {
            throw std::runtime_error("Umbilicus control_points array is empty");
        }
        return Umbilicus::FromPoints(std::move(points), volumeShape);
    }
}

static void create_output_zarr(const fs::path& outputZarr, const Layout& layout, const Config& cfg)
{
    fs::create_directories(outputZarr);
    vc::createZarrDataset(outputZarr / "x", "0", layout.outputShapeZyx, layout.chunkShapeZyx, vc::VcDtype::uint8, "blosc", "/", 128);
    vc::createZarrDataset(outputZarr / "y", "0", layout.outputShapeZyx, layout.chunkShapeZyx, vc::VcDtype::uint8, "blosc", "/", 128);
    vc::createZarrDataset(outputZarr / "z", "0", layout.outputShapeZyx, layout.chunkShapeZyx, vc::VcDtype::uint8, "blosc", "/", 128);
    vc::createZarrDataset(outputZarr / "support_count", "0", layout.outputShapeZyx, layout.chunkShapeZyx, vc::VcDtype::uint8, "blosc", "/", 0);
    vc::createZarrDataset(outputZarr / "consensus_ratio", "0", layout.outputShapeZyx, layout.chunkShapeZyx, vc::VcDtype::uint8, "blosc", "/", 0);
    vc::createZarrDataset(outputZarr / "angular_dispersion", "0", layout.outputShapeZyx, layout.chunkShapeZyx, vc::VcDtype::uint8, "blosc", "/", 255);

    auto mk3 = [](auto a, auto b, auto c) {
        Json arr = Json::array();
        arr.push_back(static_cast<int64_t>(a));
        arr.push_back(static_cast<int64_t>(b));
        arr.push_back(static_cast<int64_t>(c));
        return arr;
    };

    Json encoding = Json::object();
    encoding["type"] = "direction-field-u8";
    encoding["decode"] = "(u8-128)/127";
    encoding["fill_value"] = 128;

    Json attrs = Json::object();
    attrs["source"] = "vc_patch_normals3d";
    attrs["description"] = "Patch-derived direction-field encoded normals: x/0,y/0,z/0 uint8 with decode (u8-128)/127";
    attrs["encoding"] = encoding;
    attrs["align_normals"] = true;
    attrs["alignment_method"] = "umbilicus_radial";
    attrs["grid_origin_xyz"] = mk3(0, 0, 0);
    attrs["sample_step"] = cfg.sampleStep;
    attrs["support_radius_voxels"] = cfg.radius;
    attrs["min_support_radius_voxels"] = cfg.minRadius;
    attrs["radius_growth"] = cfg.radiusGrowth;
    attrs["target_support"] = cfg.targetSupport;
    attrs["surface_stride"] = cfg.surfaceStride;
    attrs["cluster_angle_degrees"] = cfg.clusterAngleDeg;
    attrs["dominance_ratio"] = cfg.dominanceRatio;
    attrs["min_support"] = cfg.minSupport;
    attrs["reference_zarr"] = cfg.referenceZarr.string();
    attrs["reference_level"] = cfg.level;
    attrs["umbilicus_path"] = cfg.umbilicusPath.string();
    if (cfg.cropXyz.has_value()) {
        const auto& c = *cfg.cropXyz;
        attrs["crop_min_xyz"] = mk3(c[0], c[1], c[2]);
        attrs["crop_max_xyz"] = mk3(c[3], c[4], c[5]);
    }
    attrs["grid_shape_zyx"] = mk3(layout.outputShapeZyx[0], layout.outputShapeZyx[1], layout.outputShapeZyx[2]);
    attrs["volume_shape_zyx"] = mk3(layout.volumeShapeZyx[0], layout.volumeShapeZyx[1], layout.volumeShapeZyx[2]);
    vc::writeZarrAttributes(outputZarr, attrs);
}

static ChunkResult compute_chunk(
    const SurfacePatchIndex& index,
    const Umbilicus& umbilicus,
    const Config& cfg,
    const Layout& layout,
    const std::array<size_t, 3>& off,
    const std::array<size_t, 3>& shape)
{
    const auto total_t0 = std::chrono::steady_clock::now();
    (void)total_t0;

    ChunkResult result;
    result.off = off;
    result.shape = shape;
    const size_t count = shape[0] * shape[1] * shape[2];
    result.x.assign(count, 128);
    result.y.assign(count, 128);
    result.z.assign(count, 128);
    result.support.assign(count, 0);
    result.consensus.assign(count, 0);
    result.angularDispersion.assign(count, 255);

    const std::vector<double> radii = make_radius_buckets(cfg);
    std::vector<double> radiusSqBuckets;
    radiusSqBuckets.reserve(radii.size());
    for (double r : radii) {
        radiusSqBuckets.push_back(r * r);
    }
    std::vector<CellAccumulator> cells(count * radii.size());
    const double cosThreshold = std::cos(static_cast<double>(cfg.clusterAngleDeg) * CV_PI / 180.0);
    const double radius = cfg.radius;
    const double radiusSq = radius * radius;
    const double step = static_cast<double>(cfg.sampleStep);

    const double x0 = static_cast<double>(off[2]) * step;
    const double y0 = static_cast<double>(off[1]) * step;
    const double z0 = static_cast<double>(off[0]) * step;
    const double x1 = static_cast<double>(off[2] + shape[2] - 1) * step;
    const double y1 = static_cast<double>(off[1] + shape[1] - 1) * step;
    const double z1 = static_cast<double>(off[0] + shape[0] - 1) * step;

    Rect3D bounds;
    bounds.low = cv::Vec3f(
        static_cast<float>(std::max(0.0, x0 - radius)),
        static_cast<float>(std::max(0.0, y0 - radius)),
        static_cast<float>(std::max(0.0, z0 - radius)));
    bounds.high = cv::Vec3f(
        static_cast<float>(std::min<double>(layout.volumeShapeZyx[2] - 1, x1 + radius)),
        static_cast<float>(std::min<double>(layout.volumeShapeZyx[1] - 1, y1 + radius)),
        static_cast<float>(std::min<double>(layout.volumeShapeZyx[0] - 1, z1 + radius)));

    const auto query_t0 = std::chrono::steady_clock::now();
    std::vector<SurfacePatchIndex::TriangleCandidate> triangles;
    triangles.reserve(2048);
    SurfacePatchIndex::TriangleQuery query;
    query.bounds = bounds;
    index.forEachTriangle(query, [&](const SurfacePatchIndex::TriangleCandidate& tri) {
        triangles.push_back(tri);
    });
    result.querySeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - query_t0).count();
    result.queriedTriangles = triangles.size();

    const auto bin_t0 = std::chrono::steady_clock::now();
    for (const auto& tri : triangles) {
        const cv::Vec3d a(tri.world[0][0], tri.world[0][1], tri.world[0][2]);
        const cv::Vec3d b(tri.world[1][0], tri.world[1][1], tri.world[1][2]);
        const cv::Vec3d c(tri.world[2][0], tri.world[2][1], tri.world[2][2]);
        cv::Vec3d normal = (b - a).cross(c - a);
        const double area2 = std::sqrt(normal.dot(normal));
        if (!(area2 > 1e-9)) {
            continue;
        }
        normal *= 1.0 / area2;

        const double triMinX = std::min({a[0], b[0], c[0]}) - radius;
        const double triMaxX = std::max({a[0], b[0], c[0]}) + radius;
        const double triMinY = std::min({a[1], b[1], c[1]}) - radius;
        const double triMaxY = std::max({a[1], b[1], c[1]}) + radius;
        const double triMinZ = std::min({a[2], b[2], c[2]}) - radius;
        const double triMaxZ = std::max({a[2], b[2], c[2]}) + radius;

        const int lx0 = std::max<int>(0, static_cast<int>(std::ceil(triMinX / step)) - static_cast<int>(off[2]));
        const int ly0 = std::max<int>(0, static_cast<int>(std::ceil(triMinY / step)) - static_cast<int>(off[1]));
        const int lz0 = std::max<int>(0, static_cast<int>(std::ceil(triMinZ / step)) - static_cast<int>(off[0]));
        const int lx1 = std::min<int>(static_cast<int>(shape[2]), static_cast<int>(std::floor(triMaxX / step)) - static_cast<int>(off[2]) + 1);
        const int ly1 = std::min<int>(static_cast<int>(shape[1]), static_cast<int>(std::floor(triMaxY / step)) - static_cast<int>(off[1]) + 1);
        const int lz1 = std::min<int>(static_cast<int>(shape[0]), static_cast<int>(std::floor(triMaxZ / step)) - static_cast<int>(off[0]) + 1);

        for (int lz = lz0; lz < lz1; ++lz) {
            const double wz = static_cast<double>(off[0] + static_cast<size_t>(lz)) * step;
            for (int ly = ly0; ly < ly1; ++ly) {
                const double wy = static_cast<double>(off[1] + static_cast<size_t>(ly)) * step;
                for (int lx = lx0; lx < lx1; ++lx) {
                    const double wx = static_cast<double>(off[2] + static_cast<size_t>(lx)) * step;
                    const cv::Vec3d p(wx, wy, wz);
                    const double d2 = point_triangle_distance_sq(p, a, b, c);
                    if (!(d2 <= radiusSq)) {
                        continue;
                    }

                    cv::Vec3d oriented = normal;
                    const cv::Vec3f center = umbilicus.center_at(static_cast<int>(std::lround(wz)));
                    cv::Vec3d radial(wx - center[0], wy - center[1], wz - center[2]);
                    const double rNorm = std::sqrt(radial.dot(radial));
                    if (rNorm > 1e-6) {
                        radial *= 1.0 / rNorm;
                        if (oriented.dot(radial) < 0.0) {
                            oriented = -oriented;
                        }
                    }

                    const size_t idx = (static_cast<size_t>(lz) * shape[1] + static_cast<size_t>(ly)) * shape[2] + static_cast<size_t>(lx);
                    auto firstBucket = std::lower_bound(radiusSqBuckets.begin(), radiusSqBuckets.end(), d2);
                    if (firstBucket == radiusSqBuckets.end()) {
                        continue;
                    }
                    const size_t firstBucketIdx = static_cast<size_t>(std::distance(radiusSqBuckets.begin(), firstBucket));
                    const double falloff = std::max(0.0, 1.0 - std::sqrt(std::max(0.0, d2)) / radii[firstBucketIdx]);
                    const double weight = std::max(1e-6, falloff) * std::sqrt(area2);
                    cells[idx * radii.size() + firstBucketIdx].add(oriented, weight, cosThreshold);
                    ++result.binnedObservations;
                }
            }
        }
    }
    result.binSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - bin_t0).count();

    for (size_t idx = 0; idx < count; ++idx) {
        CellAccumulator merged;
        CellAccumulator* selected = nullptr;
        for (size_t rb = 0; rb < radii.size(); ++rb) {
            const CellAccumulator& annulus = cells[idx * radii.size() + rb];
            merged.mergeFrom(annulus, cosThreshold);
            if (merged.observations >= cfg.targetSupport) {
                selected = &merged;
                break;
            }
        }
        if (selected == nullptr && !merged.clusters.empty()) {
            selected = &merged;
        }

        if (selected == nullptr || selected->clusters.empty()) {
            ++result.emptyCells;
            continue;
        }

        auto& cell = *selected;
        std::sort(cell.clusters.begin(), cell.clusters.end(), [](const Cluster& a, const Cluster& b) {
            return a.weight > b.weight;
        });
        const Cluster& best = cell.clusters.front();
        const double second = cell.clusters.size() > 1 ? cell.clusters[1].weight : 0.0;
        const double ratio = second > 1e-12 ? best.weight / second : 255.0;
        if (best.count < cfg.minSupport) {
            ++result.lowSupportCells;
            result.support[idx] = encode_count(best.count);
            result.consensus[idx] = encode_unit(std::min(1.0, ratio / std::max(1.0f, cfg.dominanceRatio)));
            continue;
        }
        if (ratio < cfg.dominanceRatio) {
            ++result.ambiguousCells;
            result.support[idx] = encode_count(best.count);
            result.consensus[idx] = encode_unit(std::min(1.0, ratio / std::max(1.0f, cfg.dominanceRatio)));
            continue;
        }

        cv::Vec3d n = best.sum;
        const double nLen = std::sqrt(n.dot(n));
        if (!(nLen > 1e-12)) {
            continue;
        }
        n *= 1.0 / nLen;
        const double dispersion = 1.0 - std::clamp(nLen / std::max(1e-12, best.weight), 0.0, 1.0);

        result.x[idx] = encode_dir_component(n[0]);
        result.y[idx] = encode_dir_component(n[1]);
        result.z[idx] = encode_dir_component(n[2]);
        result.support[idx] = encode_count(best.count);
        result.consensus[idx] = encode_unit(std::min(1.0, ratio / std::max(1.0f, cfg.dominanceRatio)));
        result.angularDispersion[idx] = encode_unit(dispersion);
        ++result.valid;
    }

    return result;
}

static void write_chunk(const fs::path& outputZarr, const ChunkResult& r)
{
    auto dsx = std::make_unique<vc::VcDataset>(outputZarr / "x" / "0");
    auto dsy = std::make_unique<vc::VcDataset>(outputZarr / "y" / "0");
    auto dsz = std::make_unique<vc::VcDataset>(outputZarr / "z" / "0");
    auto dss = std::make_unique<vc::VcDataset>(outputZarr / "support_count" / "0");
    auto dsr = std::make_unique<vc::VcDataset>(outputZarr / "consensus_ratio" / "0");
    auto dsd = std::make_unique<vc::VcDataset>(outputZarr / "angular_dispersion" / "0");
    const std::vector<size_t> off = {r.off[0], r.off[1], r.off[2]};
    const std::vector<size_t> shape = {r.shape[0], r.shape[1], r.shape[2]};
    writeZarrRegionU8ByChunk(dsx.get(), off, shape, r.x.data(), 128);
    writeZarrRegionU8ByChunk(dsy.get(), off, shape, r.y.data(), 128);
    writeZarrRegionU8ByChunk(dsz.get(), off, shape, r.z.data(), 128);
    writeZarrRegionU8ByChunk(dss.get(), off, shape, r.support.data(), 0);
    writeZarrRegionU8ByChunk(dsr.get(), off, shape, r.consensus.data(), 0);
    writeZarrRegionU8ByChunk(dsd.get(), off, shape, r.angularDispersion.data(), 255);
}

static int run(const Config& cfg)
{
    const auto totalStart = std::chrono::steady_clock::now();
    Config effective = cfg;
    if (!(effective.radius > 0.0f)) {
        effective.radius = 192.0f;
    }
    effective.minRadius = std::clamp(effective.minRadius, 1.0f, effective.radius);
    effective.radiusGrowth = std::max(1.01f, effective.radiusGrowth);

    std::cout << "Patch normals3d generator\n"
              << "  patches: " << effective.patchesDir << "\n"
              << "  reference: " << effective.referenceZarr << " level " << effective.level << "\n"
              << "  output: " << effective.outputZarr << "\n"
              << "  step/radius: " << effective.sampleStep << " / " << effective.minRadius << ".." << effective.radius
              << " target_support=" << effective.targetSupport << "\n"
              << "  OpenMP threads: " << std::max(1, omp_get_max_threads()) << "\n";

    Layout layout = make_layout(effective.referenceZarr, effective.level, effective.sampleStep, effective.chunkSize);
    std::cout << "  volume shape ZYX: " << layout.volumeShapeZyx[0] << " x " << layout.volumeShapeZyx[1] << " x " << layout.volumeShapeZyx[2] << "\n"
              << "  output shape ZYX: " << layout.outputShapeZyx[0] << " x " << layout.outputShapeZyx[1] << " x " << layout.outputShapeZyx[2] << "\n";

    auto surfaces = discover_surfaces(effective.patchesDir, effective.previewLimit, effective.cropXyz);
    if (surfaces.empty()) {
        throw std::runtime_error("No tifxyz patches found under: " + effective.patchesDir.string());
    }
    std::cout << "Discovered " << surfaces.size() << " tifxyz patches\n";

    SurfacePatchIndex index;
    index.setSamplingStride(effective.surfaceStride);
    const auto indexStart = std::chrono::steady_clock::now();
    std::cout << "Building SurfacePatchIndex...\n";
    index.rebuild(surfaces, effective.radius);
    const double indexSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - indexStart).count();
    std::cout << "SurfacePatchIndex built: surfaces=" << index.surfaceCount()
              << " patches=" << index.patchCount()
              << " seconds=" << indexSeconds << "\n";

    Umbilicus umbilicus = load_umbilicus_for_volume(effective.umbilicusPath, layout.volumeShapeZyx);

    std::error_code ec;
    fs::remove_all(effective.outputZarr, ec);
    create_output_zarr(effective.outputZarr, layout, effective);

    std::array<size_t, 3> outBegin{0, 0, 0};
    std::array<size_t, 3> outEnd{layout.outputShapeZyx[0], layout.outputShapeZyx[1], layout.outputShapeZyx[2]};
    if (effective.cropXyz.has_value()) {
        const auto& c = *effective.cropXyz;
        const auto clamp_coord = [](int v, size_t hi) -> size_t {
            return static_cast<size_t>(std::clamp(v, 0, static_cast<int>(hi)));
        };
        const size_t x0 = clamp_coord(c[0], layout.volumeShapeZyx[2]);
        const size_t y0 = clamp_coord(c[1], layout.volumeShapeZyx[1]);
        const size_t z0 = clamp_coord(c[2], layout.volumeShapeZyx[0]);
        const size_t x1 = clamp_coord(c[3], layout.volumeShapeZyx[2]);
        const size_t y1 = clamp_coord(c[4], layout.volumeShapeZyx[1]);
        const size_t z1 = clamp_coord(c[5], layout.volumeShapeZyx[0]);
        if (x1 <= x0 || y1 <= y0 || z1 <= z0) {
            throw std::runtime_error("--crop maps to an empty region");
        }
        const size_t step = static_cast<size_t>(effective.sampleStep);
        outBegin = {z0 / step, y0 / step, x0 / step};
        outEnd = {
            std::min(layout.outputShapeZyx[0], (z1 + step - 1) / step),
            std::min(layout.outputShapeZyx[1], (y1 + step - 1) / step),
            std::min(layout.outputShapeZyx[2], (x1 + step - 1) / step),
        };
        std::cout << "  crop output ZYX: [" << outBegin[0] << "," << outBegin[1] << "," << outBegin[2]
                  << "] -> [" << outEnd[0] << "," << outEnd[1] << "," << outEnd[2] << "]\n";
    }

    std::vector<std::array<size_t, 3>> chunkOffsets;
    for (size_t z = outBegin[0]; z < outEnd[0]; z += layout.chunkShapeZyx[0]) {
        for (size_t y = outBegin[1]; y < outEnd[1]; y += layout.chunkShapeZyx[1]) {
            for (size_t x = outBegin[2]; x < outEnd[2]; x += layout.chunkShapeZyx[2]) {
                chunkOffsets.push_back({z, y, x});
            }
        }
    }

    std::atomic<size_t> done{0};
    std::atomic<uint64_t> validTotal{0};
    std::atomic<uint64_t> emptyTotal{0};
    std::atomic<uint64_t> lowSupportTotal{0};
    std::atomic<uint64_t> ambiguousTotal{0};
    std::atomic<uint64_t> trianglesTotal{0};
    std::atomic<uint64_t> observationsTotal{0};
    const auto workStart = std::chrono::steady_clock::now();

    #pragma omp parallel for schedule(dynamic)
    for (size_t ci = 0; ci < chunkOffsets.size(); ++ci) {
        const auto off = chunkOffsets[ci];
        const std::array<size_t, 3> shape = {
            std::min(layout.chunkShapeZyx[0], outEnd[0] - off[0]),
            std::min(layout.chunkShapeZyx[1], outEnd[1] - off[1]),
            std::min(layout.chunkShapeZyx[2], outEnd[2] - off[2]),
        };
        ChunkResult r = compute_chunk(index, umbilicus, effective, layout, off, shape);
        #pragma omp critical(vc_patch_normals3d_write)
        {
            write_chunk(effective.outputZarr, r);
        }
        validTotal.fetch_add(r.valid, std::memory_order_relaxed);
        emptyTotal.fetch_add(r.emptyCells, std::memory_order_relaxed);
        lowSupportTotal.fetch_add(r.lowSupportCells, std::memory_order_relaxed);
        ambiguousTotal.fetch_add(r.ambiguousCells, std::memory_order_relaxed);
        trianglesTotal.fetch_add(r.queriedTriangles, std::memory_order_relaxed);
        observationsTotal.fetch_add(r.binnedObservations, std::memory_order_relaxed);
        const size_t finished = done.fetch_add(1, std::memory_order_relaxed) + 1;
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - workStart).count();
        if (finished == chunkOffsets.size() || finished % 10 == 0) {
            #pragma omp critical(vc_patch_normals3d_progress)
            {
                const double rate = elapsed > 1e-9 ? static_cast<double>(finished) / elapsed : 0.0;
                const double eta = rate > 1e-9 ? static_cast<double>(chunkOffsets.size() - finished) / rate : 0.0;
                std::cout << "chunks " << finished << "/" << chunkOffsets.size()
                          << " (" << std::fixed << std::setprecision(1)
                          << (100.0 * static_cast<double>(finished) / static_cast<double>(chunkOffsets.size())) << "%)"
                          << " valid=" << validTotal.load(std::memory_order_relaxed)
                          << " empty=" << emptyTotal.load(std::memory_order_relaxed)
                          << " low_support=" << lowSupportTotal.load(std::memory_order_relaxed)
                          << " ambiguous=" << ambiguousTotal.load(std::memory_order_relaxed)
                          << " triangles=" << trianglesTotal.load(std::memory_order_relaxed)
                          << " observations=" << observationsTotal.load(std::memory_order_relaxed)
                          << " elapsed=" << elapsed << "s"
                          << " eta=" << eta << "s\n";
            }
        }
    }

    const double totalSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - totalStart).count();
    std::cout << "Complete. valid normals=" << validTotal.load()
              << " empty=" << emptyTotal.load()
              << " low_support=" << lowSupportTotal.load()
              << " ambiguous=" << ambiguousTotal.load()
              << " chunks=" << chunkOffsets.size()
              << " total seconds=" << totalSeconds << "\n";

    if (effective.metricsJson.has_value()) {
        Json metrics = Json::object();
        metrics["mode"] = "patch-normals3d";
        metrics["patches"] = effective.patchesDir.string();
        metrics["reference_zarr"] = effective.referenceZarr.string();
        metrics["output_zarr"] = effective.outputZarr.string();
        metrics["umbilicus_path"] = effective.umbilicusPath.string();
        metrics["patch_count"] = surfaces.size();
        metrics["surface_index_patches"] = index.patchCount();
        metrics["index_seconds"] = indexSeconds;
        metrics["chunks"] = chunkOffsets.size();
        metrics["valid_normals"] = static_cast<int64_t>(validTotal.load());
        metrics["empty_cells"] = static_cast<int64_t>(emptyTotal.load());
        metrics["low_support_cells"] = static_cast<int64_t>(lowSupportTotal.load());
        metrics["ambiguous_cells"] = static_cast<int64_t>(ambiguousTotal.load());
        metrics["queried_triangles"] = static_cast<int64_t>(trianglesTotal.load());
        metrics["binned_observations"] = static_cast<int64_t>(observationsTotal.load());
        metrics["total_seconds"] = totalSeconds;
        std::ofstream out(*effective.metricsJson);
        out << metrics.dump(2) << "\n";
    }
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    po::options_description desc("vc_patch_normals3d options");
    desc.add_options()
        ("help,h", "Print help")
        ("patches,p", po::value<std::string>(), "Directory containing tifxyz patch subdirectories")
        ("reference-zarr,r", po::value<std::string>(), "Reference OME-Zarr volume root")
        ("umbilicus,u", po::value<std::string>(), "Umbilicus JSON path")
        ("output-zarr,o", po::value<std::string>(), "Output normal3d Zarr root")
        ("level", po::value<int>()->default_value(0), "Reference Zarr level")
        ("step", po::value<int>()->default_value(16), "Output sample step in voxels")
        ("radius", po::value<float>()->default_value(192.0f), "Max adaptive support radius in voxels")
        ("min-radius", po::value<float>()->default_value(32.0f), "Initial adaptive support radius in voxels")
        ("radius-growth", po::value<float>()->default_value(1.5f), "Adaptive radius multiplier")
        ("target-support", po::value<int>()->default_value(256), "Stop radius growth after N observations")
        ("crop", po::value<std::vector<int>>()->multitoken(), "Voxel crop x0 y0 z0 x1 y1 z1, half-open")
        ("chunk-size", po::value<int>()->default_value(64), "Output chunk edge in samples")
        ("surface-stride", po::value<int>()->default_value(1), "Surface triangulation stride")
        ("cluster-angle", po::value<float>()->default_value(25.0f), "Sign-invariant cluster angle in degrees")
        ("dominance-ratio", po::value<float>()->default_value(1.0f), "Best/second cluster dominance ratio")
        ("min-support", po::value<int>()->default_value(1), "Minimum winning-cluster observation count")
        ("limit", po::value<int>()->default_value(0), "Load only first N patches")
        ("metrics-json", po::value<std::string>(), "Write metrics JSON");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            print_usage();
            std::cout << "\n" << desc << "\n";
            return 0;
        }
        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << "\n\n" << desc << "\n";
        return 1;
    }

    for (const char* required : {"patches", "reference-zarr", "umbilicus", "output-zarr"}) {
        if (!vm.count(required)) {
            std::cerr << "Error: missing required --" << required << "\n\n";
            print_usage();
            return 1;
        }
    }

    Config cfg;
    cfg.patchesDir = vm["patches"].as<std::string>();
    cfg.referenceZarr = vm["reference-zarr"].as<std::string>();
    cfg.umbilicusPath = vm["umbilicus"].as<std::string>();
    cfg.outputZarr = vm["output-zarr"].as<std::string>();
    cfg.level = vm["level"].as<int>();
    cfg.sampleStep = std::max(1, vm["step"].as<int>());
    cfg.radius = vm["radius"].as<float>();
    cfg.minRadius = vm["min-radius"].as<float>();
    cfg.radiusGrowth = vm["radius-growth"].as<float>();
    cfg.targetSupport = std::max(1, vm["target-support"].as<int>());
    cfg.chunkSize = std::max(1, vm["chunk-size"].as<int>());
    cfg.surfaceStride = std::max(1, vm["surface-stride"].as<int>());
    cfg.clusterAngleDeg = std::max(1.0f, vm["cluster-angle"].as<float>());
    cfg.dominanceRatio = std::max(1.0f, vm["dominance-ratio"].as<float>());
    cfg.minSupport = std::max(1, vm["min-support"].as<int>());
    cfg.previewLimit = std::max(0, vm["limit"].as<int>());
    if (vm.count("crop")) {
        const auto values = vm["crop"].as<std::vector<int>>();
        if (values.size() != 6) {
            std::cerr << "Error: --crop requires exactly 6 integers: x0 y0 z0 x1 y1 z1\n";
            return 1;
        }
        cfg.cropXyz = std::array<int, 6>{values[0], values[1], values[2], values[3], values[4], values[5]};
    }
    if (vm.count("metrics-json")) {
        cfg.metricsJson = fs::path(vm["metrics-json"].as<std::string>());
    }

    try {
        return run(cfg);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
