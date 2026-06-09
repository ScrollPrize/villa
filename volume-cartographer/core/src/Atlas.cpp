#include "vc/atlas/Atlas.hpp"

#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/lasagna/Manifest.hpp"
#include "vc/lasagna/LineModel.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace fs = std::filesystem;

namespace vc::atlas {
namespace {

constexpr double kEpsilon = 1.0e-9;
constexpr double kControlPointMatchEpsilon = 1.0e-8;
constexpr int kAtlasMetadataVersion = 4;
constexpr int kFiberMappingVersion = 3;

bool atlasDebugEnabled()
{
    const char* value = std::getenv("VC_ATLAS_DEBUG");
    return value && *value != '\0' && std::string_view(value) != "0";
}

void atlasDebug(const std::string& message)
{
    if (atlasDebugEnabled()) {
        std::cerr << "[atlas] " << message << std::endl;
    }
}

std::string vecString(const cv::Vec3d& p)
{
    std::ostringstream out;
    out << '(' << p[0] << ", " << p[1] << ", " << p[2] << ')';
    return out.str();
}

bool finitePoint(const cv::Vec3d& p)
{
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

bool finitePoint(const cv::Vec3f& p)
{
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

double squaredDistance(const cv::Vec3d& a, const cv::Vec3d& b)
{
    const cv::Vec3d d = a - b;
    return d.dot(d);
}

double distance(const cv::Vec3d& a, const cv::Vec3d& b)
{
    return std::sqrt(squaredDistance(a, b));
}

double normalizeAtlasU(double atlasU, int periodColumns)
{
    if (periodColumns <= 0 || !std::isfinite(atlasU)) {
        return atlasU;
    }
    const double period = static_cast<double>(periodColumns);
    double normalized = std::fmod(atlasU, period);
    if (normalized < 0.0) {
        normalized += period;
    }
    if (normalized >= period) {
        normalized -= period;
    }
    return normalized;
}

cv::Vec3d toVec3d(const cv::Vec3f& p)
{
    return {p[0], p[1], p[2]};
}

cv::Vec3f toVec3f(const cv::Vec3d& p)
{
    return {static_cast<float>(p[0]),
            static_cast<float>(p[1]),
            static_cast<float>(p[2])};
}

nlohmann::json pointJson(const cv::Vec3d& p)
{
    return nlohmann::json::array({p[0], p[1], p[2]});
}

cv::Vec3d pointFromJson(const nlohmann::json& value)
{
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error("atlas point must be a 3-number array");
    }
    cv::Vec3d p{value.at(0).get<double>(),
                value.at(1).get<double>(),
                value.at(2).get<double>()};
    if (!finitePoint(p)) {
        throw std::runtime_error("atlas point contains non-finite coordinates");
    }
    return p;
}

nlohmann::json anchorJson(const AtlasAnchor& anchor)
{
    return {
        {"source_index", anchor.sourceIndex},
        {"world", pointJson(anchor.world)},
        {"atlas", nlohmann::json::array({anchor.atlasU, anchor.atlasV})},
        {"distance", anchor.distance},
    };
}

bool finiteAtlasCoord(double u, double v)
{
    return std::isfinite(u) && std::isfinite(v);
}

nlohmann::json linkEndpointJson(const AtlasLinkEndpoint& endpoint)
{
    return {
        {"object_type", "fiber"},
        {"fiber_path", endpoint.fiberPath.generic_string()},
        {"source_index", endpoint.sourceIndex},
        {"arclength", endpoint.arclength},
        {"base_atlas", nlohmann::json::array({endpoint.atlasU, endpoint.atlasV})},
    };
}

AtlasLinkEndpoint linkEndpointFromJson(const nlohmann::json& value)
{
    if (!value.is_object()) {
        throw std::runtime_error("atlas link endpoint must be an object");
    }
    if (value.value("object_type", std::string{"fiber"}) != "fiber") {
        throw std::runtime_error("atlas link endpoint object_type must be fiber");
    }
    AtlasLinkEndpoint endpoint;
    endpoint.fiberPath = value.at("fiber_path").get<std::string>();
    endpoint.sourceIndex = value.value("source_index", 0);
    endpoint.arclength = value.value("arclength", 0.0);
    const auto& atlas = value.at("base_atlas");
    if (!atlas.is_array() || atlas.size() != 2) {
        throw std::runtime_error("atlas link endpoint must contain base_atlas [u, v]");
    }
    endpoint.atlasU = atlas.at(0).get<double>();
    endpoint.atlasV = atlas.at(1).get<double>();
    if (endpoint.fiberPath.empty() ||
        !finiteAtlasCoord(endpoint.atlasU, endpoint.atlasV) ||
        !std::isfinite(endpoint.arclength)) {
        throw std::runtime_error("atlas link endpoint contains invalid values");
    }
    return endpoint;
}

nlohmann::json linkJson(const AtlasLink& link)
{
    return {
        {"first", linkEndpointJson(link.first)},
        {"second", linkEndpointJson(link.second)},
        {"desired_winding_delta", link.desiredWindingDelta},
    };
}

AtlasLink linkFromJson(const nlohmann::json& value)
{
    if (!value.is_object()) {
        throw std::runtime_error("atlas link must be an object");
    }
    AtlasLink link;
    link.first = linkEndpointFromJson(value.at("first"));
    link.second = linkEndpointFromJson(value.at("second"));
    link.desiredWindingDelta = value.at("desired_winding_delta").get<int>();
    return link;
}

AtlasAnchor anchorFromJson(const nlohmann::json& value)
{
    AtlasAnchor anchor;
    anchor.sourceIndex = value.at("source_index").get<int>();
    anchor.world = pointFromJson(value.at("world"));
    const auto& atlas = value.at("atlas");
    if (!atlas.is_array() || atlas.size() != 2) {
        throw std::runtime_error("atlas anchor must contain [u, v]");
    }
    anchor.atlasU = atlas.at(0).get<double>();
    anchor.atlasV = atlas.at(1).get<double>();
    anchor.distance = value.value("distance", 0.0);
    return anchor;
}

void writeJsonFile(const fs::path& path, const nlohmann::json& json)
{
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    if (ec) {
        throw std::runtime_error("failed to create " + path.parent_path().string() + ": " + ec.message());
    }
    const fs::path tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp);
        if (!out) {
            throw std::runtime_error("failed to open " + tmp.string());
        }
        out << json.dump(2) << '\n';
    }
    fs::rename(tmp, path, ec);
    if (ec) {
        fs::remove(tmp);
        throw std::runtime_error("failed to replace " + path.string() + ": " + ec.message());
    }
}

nlohmann::json readJsonFile(const fs::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open " + path.string());
    }
    return nlohmann::json::parse(in);
}

std::vector<cv::Vec3d> pointArrayFromJson(const nlohmann::json& root,
                                          const char* key,
                                          const fs::path& path)
{
    const auto it = root.find(key);
    if (it == root.end() || !it->is_array()) {
        throw std::runtime_error("fiber JSON is missing array " + std::string(key) +
                                 ": " + path.string());
    }
    std::vector<cv::Vec3d> points;
    points.reserve(it->size());
    for (const auto& point : *it) {
        points.push_back(pointFromJson(point));
    }
    return points;
}

FiberInput loadSourceFiberInput(const fs::path& fiberPath,
                                const fs::path& fiberRelativePath)
{
    const auto root = readJsonFile(fiberPath);
    if (root.value("type", std::string{}) != "vc3d_fiber") {
        throw std::runtime_error("fiber JSON is not a vc3d_fiber: " + fiberPath.string());
    }
    if (root.value("version", 0) != 1) {
        throw std::runtime_error("unsupported vc3d_fiber version in " + fiberPath.string());
    }
    FiberInput input;
    input.fiberPath = fiberRelativePath;
    input.controlPoints = pointArrayFromJson(root, "control_points", fiberPath);
    input.linePoints = pointArrayFromJson(root, "line_points", fiberPath);
    validateFiberInputControlPoints(input);
    return input;
}

fs::path inferVolpkgRootFromAtlasDir(const fs::path& atlasDir)
{
    if (atlasDir.parent_path().filename() == "atlases") {
        return atlasDir.parent_path().parent_path();
    }
    return {};
}

fs::path resolveAtlasRelativePath(const fs::path& atlasDir,
                                  const fs::path& volpkgRoot,
                                  const fs::path& jsonPath)
{
    if (jsonPath.empty()) {
        return {};
    }
    if (jsonPath.is_absolute()) {
        return jsonPath;
    }
    if (!volpkgRoot.empty()) {
        return (volpkgRoot / jsonPath).lexically_normal();
    }
    return (atlasDir / jsonPath).lexically_normal();
}

std::vector<fs::path> sortedAtlasFiberMappingFiles(const fs::path& atlasDir)
{
    const fs::path mappingsDir = atlasDir / "mappings" / "fibers";
    std::vector<fs::path> mappingFiles;
    if (!fs::is_directory(mappingsDir)) {
        return mappingFiles;
    }
    for (const auto& entry : fs::directory_iterator(mappingsDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".json") {
            mappingFiles.push_back(entry.path());
        }
    }
    std::sort(mappingFiles.begin(), mappingFiles.end());
    return mappingFiles;
}

Atlas loadAtlasContextForRebuild(const fs::path& atlasDir)
{
    Atlas atlas;
    const auto metadata = readJsonFile(atlasDir / "metadata.json");
    atlas.metadata.type = metadata.value("type", std::string{});
    atlas.metadata.version = metadata.value("version", 0);
    if (atlas.metadata.type != "vc3d_atlas") {
        throw std::runtime_error("unsupported atlas metadata in " + atlasDir.string());
    }
    if (metadata.contains("idx_rotation_columns") || !metadata.contains("zero_winding_column")) {
        throw std::runtime_error("unsupported atlas metadata in " + atlasDir.string());
    }
    atlas.metadata.name = metadata.value("name", atlasDir.filename().string());
    atlas.metadata.baseMeshPath = metadata.value("base_mesh_path", std::string{});
    atlas.metadata.sourceBaseMeshPath = metadata.value("source_base_mesh_path", std::string{});
    atlas.metadata.zeroWindingColumn = metadata.at("zero_winding_column").get<int>();
    atlas.metadata.seedLineIndex = metadata.value("seed_line_index", 0);
    if (metadata.contains("seed_atlas") && metadata["seed_atlas"].is_array() &&
        metadata["seed_atlas"].size() == 2) {
        atlas.metadata.seedAtlasU = metadata["seed_atlas"][0].get<double>();
        atlas.metadata.seedAtlasV = metadata["seed_atlas"][1].get<double>();
    }

    const fs::path linksPath = atlasDir / "links.json";
    if (fs::exists(linksPath)) {
        const auto linksJson = readJsonFile(linksPath);
        if (linksJson.contains("links") && linksJson["links"].is_array()) {
            for (const auto& link : linksJson["links"]) {
                if (link.is_string()) {
                    continue;
                }
                atlas.links.push_back(linkFromJson(link));
            }
        }
    }

    for (const auto& mappingPath : sortedAtlasFiberMappingFiles(atlasDir)) {
        const auto root = readJsonFile(mappingPath);
        FiberMapping mapping;
        mapping.fiberPath = root.value("fiber_path", std::string{});
        mapping.windingOffset = 0;
        for (const auto& anchor : root.value("line_anchors", nlohmann::json::array())) {
            mapping.lineAnchors.push_back(anchorFromJson(anchor));
        }
        for (const auto& anchor : root.value("control_anchors", nlohmann::json::array())) {
            mapping.controlAnchors.push_back(anchorFromJson(anchor));
        }
        atlas.fibers.push_back(std::move(mapping));
    }
    return atlas;
}

int seedLineIndexForFiber(const FiberInput& fiber)
{
    if (fiber.linePoints.empty()) {
        throw std::runtime_error("fiber has no line points");
    }
    return static_cast<int>((fiber.linePoints.size() - 1) / 2);
}

struct BilinearRayHit {
    cv::Vec3d world{0.0, 0.0, 0.0};
    double u = 0.0;
    double v = 0.0;
    double t = 0.0;
};

double norm(const cv::Vec3d& v)
{
    return std::sqrt(v.dot(v));
}

std::array<cv::Vec3d, 2> perpendicularBasis(const cv::Vec3d& direction)
{
    const cv::Vec3d axis = std::abs(direction[0]) < 0.9
        ? cv::Vec3d{1.0, 0.0, 0.0}
        : cv::Vec3d{0.0, 1.0, 0.0};
    cv::Vec3d e1 = direction.cross(axis);
    const double e1Norm = norm(e1);
    if (e1Norm <= kEpsilon) {
        return {cv::Vec3d{0.0, 1.0, 0.0}, cv::Vec3d{0.0, 0.0, 1.0}};
    }
    e1 *= 1.0 / e1Norm;
    cv::Vec3d e2 = direction.cross(e1);
    const double e2Norm = norm(e2);
    if (e2Norm > kEpsilon) {
        e2 *= 1.0 / e2Norm;
    }
    return {e1, e2};
}

std::vector<double> solveQuadratic(double a, double b, double c)
{
    std::vector<double> roots;
    if (std::abs(a) <= kEpsilon) {
        if (std::abs(b) > kEpsilon) {
            roots.push_back(-c / b);
        }
        return roots;
    }

    const double discriminant = b * b - 4.0 * a * c;
    if (discriminant < -kEpsilon) {
        return roots;
    }
    if (std::abs(discriminant) <= kEpsilon) {
        roots.push_back(-b / (2.0 * a));
        return roots;
    }
    const double sqrtD = std::sqrt(discriminant);
    roots.push_back((-b - sqrtD) / (2.0 * a));
    roots.push_back((-b + sqrtD) / (2.0 * a));
    return roots;
}

cv::Vec3d bilinearPoint(const std::array<cv::Vec3d, 4>& quad, double u, double v)
{
    return quad[0] * ((1.0 - u) * (1.0 - v)) +
           quad[1] * (u * (1.0 - v)) +
           quad[2] * (u * v) +
           quad[3] * ((1.0 - u) * v);
}

std::vector<BilinearRayHit> rayBilinearQuadIntersections(
    const cv::Vec3d& origin,
    const cv::Vec3d& direction,
    double maxT,
    const std::array<cv::Vec3d, 4>& quad)
{
    const auto basis = perpendicularBasis(direction);
    const cv::Vec3d a3 = quad[0] - origin;
    const cv::Vec3d b3 = quad[1] - quad[0];
    const cv::Vec3d c3 = quad[3] - quad[0];
    const cv::Vec3d e3 = quad[0] - quad[1] - quad[3] + quad[2];

    const auto project = [&](const cv::Vec3d& p) {
        return cv::Vec2d{p.dot(basis[0]), p.dot(basis[1])};
    };

    const cv::Vec2d a = project(a3);
    const cv::Vec2d b = project(b3);
    const cv::Vec2d c = project(c3);
    const cv::Vec2d e = project(e3);

    std::vector<std::pair<double, double>> candidates;
    auto addCandidate = [&](double u, double v) {
        if (!std::isfinite(u) || !std::isfinite(v)) {
            return;
        }
        candidates.emplace_back(u, v);
    };

    const double u2 = b[1] * e[0] - e[1] * b[0];
    const double u1 = a[1] * e[0] + b[1] * c[0] - c[1] * b[0] - e[1] * a[0];
    const double u0 = a[1] * c[0] - c[1] * a[0];
    for (double u : solveQuadratic(u2, u1, u0)) {
        const double d0 = c[0] + e[0] * u;
        const double d1 = c[1] + e[1] * u;
        if (std::abs(d0) >= std::abs(d1) && std::abs(d0) > kEpsilon) {
            addCandidate(u, -(a[0] + b[0] * u) / d0);
        } else if (std::abs(d1) > kEpsilon) {
            addCandidate(u, -(a[1] + b[1] * u) / d1);
        }
    }

    const double v2 = c[1] * e[0] - e[1] * c[0];
    const double v1 = a[1] * e[0] + c[1] * b[0] - b[1] * c[0] - e[1] * a[0];
    const double v0 = a[1] * b[0] - b[1] * a[0];
    for (double v : solveQuadratic(v2, v1, v0)) {
        const double d0 = b[0] + e[0] * v;
        const double d1 = b[1] + e[1] * v;
        if (std::abs(d0) >= std::abs(d1) && std::abs(d0) > kEpsilon) {
            addCandidate(-(a[0] + c[0] * v) / d0, v);
        } else if (std::abs(d1) > kEpsilon) {
            addCandidate(-(a[1] + c[1] * v) / d1, v);
        }
    }

    std::vector<BilinearRayHit> hits;
    for (auto [uRaw, vRaw] : candidates) {
        if (uRaw < -1.0e-7 || uRaw > 1.0 + 1.0e-7 ||
            vRaw < -1.0e-7 || vRaw > 1.0 + 1.0e-7) {
            continue;
        }
        const double u = std::clamp(uRaw, 0.0, 1.0);
        const double v = std::clamp(vRaw, 0.0, 1.0);
        const cv::Vec3d world = bilinearPoint(quad, u, v);
        const double t = (world - origin).dot(direction);
        if (t < -1.0e-7 || t > maxT + 1.0e-7) {
            continue;
        }
        const cv::Vec3d rayWorld = origin + direction * t;
        const double residual = norm(world - rayWorld);
        const double scale = std::max({1.0,
                                       norm(quad[1] - quad[0]),
                                       norm(quad[2] - quad[1]),
                                       norm(quad[3] - quad[2]),
                                       norm(quad[0] - quad[3])});
        if (residual > scale * 1.0e-6) {
            continue;
        }
        const auto duplicate = std::find_if(hits.begin(), hits.end(), [&](const BilinearRayHit& hit) {
            return std::abs(hit.u - u) <= 1.0e-7 &&
                   std::abs(hit.v - v) <= 1.0e-7 &&
                   std::abs(hit.t - t) <= 1.0e-7;
        });
        if (duplicate == hits.end()) {
            hits.push_back({world, u, v, t});
        }
    }
    return hits;
}

std::array<cv::Vec3d, 4> quadCornersForCandidate(const SurfacePatchIndex::TriangleCandidate& tri,
                                                 const QuadSurface& surface,
                                                 int& col0,
                                                 int& row0,
                                                 int& col1,
                                                 int& row1)
{
    const auto* points = surface.rawPointsPtr();
    if (!points) {
        throw std::runtime_error("surface has no point grid");
    }

    double minCol = std::numeric_limits<double>::infinity();
    double maxCol = -std::numeric_limits<double>::infinity();
    double minRow = std::numeric_limits<double>::infinity();
    double maxRow = -std::numeric_limits<double>::infinity();
    for (const auto& param : tri.surfaceParams) {
        const cv::Vec2f grid = surface.ptrToGrid(param);
        minCol = std::min(minCol, static_cast<double>(grid[0]));
        maxCol = std::max(maxCol, static_cast<double>(grid[0]));
        minRow = std::min(minRow, static_cast<double>(grid[1]));
        maxRow = std::max(maxRow, static_cast<double>(grid[1]));
    }

    col0 = static_cast<int>(std::llround(minCol));
    col1 = static_cast<int>(std::llround(maxCol));
    row0 = static_cast<int>(std::llround(minRow));
    row1 = static_cast<int>(std::llround(maxRow));
    if (col0 < 0 || row0 < 0 || col1 >= points->cols || row1 >= points->rows ||
        col0 >= col1 || row0 >= row1) {
        throw std::runtime_error("surface patch candidate is outside the point grid");
    }

    const cv::Vec3f p00 = (*points)(row0, col0);
    const cv::Vec3f p10 = (*points)(row0, col1);
    const cv::Vec3f p11 = (*points)(row1, col1);
    const cv::Vec3f p01 = (*points)(row1, col0);
    return {toVec3d(p00), toVec3d(p10), toVec3d(p11), toVec3d(p01)};
}

bool validNormal(const cv::Vec3d& normal)
{
    return finitePoint(normal) && norm(normal) > kEpsilon;
}

double boundedRayHalfLength(const cv::Vec3d& linePoint,
                            const cv::Vec3d& normal,
                            const std::vector<SurfaceCandidate>& surfaces,
                            double initialRayHalfLength)
{
    double out = std::isfinite(initialRayHalfLength) && initialRayHalfLength > 0.0
        ? initialRayHalfLength
        : 1.0;
    if (!validNormal(normal)) {
        return out;
    }

    const cv::Vec3d dir = normal * (1.0 / norm(normal));
    for (const auto& candidate : surfaces) {
        if (!candidate.surface) {
            continue;
        }

        Rect3D bbox;
        try {
            bbox = candidate.surface->bbox();
        } catch (...) {
            continue;
        }
        if (!finitePoint(bbox.low) || !finitePoint(bbox.high)) {
            continue;
        }

        for (int x = 0; x < 2; ++x) {
            for (int y = 0; y < 2; ++y) {
                for (int z = 0; z < 2; ++z) {
                    const cv::Vec3d corner{
                        x ? static_cast<double>(bbox.high[0]) : static_cast<double>(bbox.low[0]),
                        y ? static_cast<double>(bbox.high[1]) : static_cast<double>(bbox.low[1]),
                        z ? static_cast<double>(bbox.high[2]) : static_cast<double>(bbox.low[2]),
                    };
                    out = std::max(out, std::abs((corner - linePoint).dot(dir)) + 16.0);
                }
            }
        }
    }
    return out;
}

std::vector<ProjectionHit> projectPointToSurfaces(const cv::Vec3d& linePoint,
                                                  const cv::Vec3d& normal,
                                                  const std::vector<SurfaceCandidate>& surfaces,
                                                  const SurfacePatchIndex& index,
                                                  double rayHalfLength)
{
    if (!validNormal(normal) || !std::isfinite(rayHalfLength) || rayHalfLength <= 0.0) {
        return {};
    }
    std::unordered_set<SurfacePatchIndex::SurfacePtr> include;
    include.reserve(surfaces.size());
    std::unordered_map<QuadSurface*, int> surfaceIndexByPtr;
    surfaceIndexByPtr.reserve(surfaces.size());
    for (int i = 0; i < static_cast<int>(surfaces.size()); ++i) {
        if (!surfaces[i].surface) {
            continue;
        }
        include.insert(surfaces[i].surface);
        surfaceIndexByPtr.emplace(surfaces[i].surface.get(), i);
    }
    if (include.empty()) {
        return {};
    }

    const double n = norm(normal);
    const cv::Vec3d dir = normal * (1.0 / n);
    const cv::Vec3d src = linePoint - dir * rayHalfLength;
    const cv::Vec3d end = linePoint + dir * rayHalfLength;
    const cv::Vec3d segment = end - src;
    const double segmentLength = norm(segment);
    if (segmentLength <= kEpsilon) {
        return {};
    }
    const cv::Vec3d rayDir = segment * (1.0 / segmentLength);

    SurfacePatchIndex::RayQuery query;
    query.src = toVec3f(src);
    query.end = toVec3f(end);
    query.minT = 0.0f;
    query.bboxPadding = 1.0f;
    query.surfaces.include = &include;

    struct VisitedCell {
        QuadSurface* surface = nullptr;
        int col0 = 0;
        int row0 = 0;
        int col1 = 0;
        int row1 = 0;
    };
    struct VisitedHash {
        size_t operator()(const VisitedCell& cell) const
        {
            size_t h = std::hash<QuadSurface*>{}(cell.surface);
            auto combine = [&h](int value) {
                h ^= std::hash<int>{}(value) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            };
            combine(cell.col0);
            combine(cell.row0);
            combine(cell.col1);
            combine(cell.row1);
            return h;
        }
    };
    struct VisitedEq {
        bool operator()(const VisitedCell& a, const VisitedCell& b) const
        {
            return a.surface == b.surface &&
                   a.col0 == b.col0 &&
                   a.row0 == b.row0 &&
                   a.col1 == b.col1 &&
                   a.row1 == b.row1;
        }
    };

    std::unordered_set<VisitedCell, VisitedHash, VisitedEq> visited;
    std::vector<ProjectionHit> hits;
    index.forEachTriangle(query, [&](const SurfacePatchIndex::TriangleCandidate& tri) {
        if (!tri.surface) {
            return;
        }
        int col0 = 0;
        int row0 = 0;
        int col1 = 0;
        int row1 = 0;
        std::array<cv::Vec3d, 4> quad;
        try {
            quad = quadCornersForCandidate(tri, *tri.surface, col0, row0, col1, row1);
        } catch (...) {
            return;
        }
        const VisitedCell cell{tri.surface.get(), col0, row0, col1, row1};
        if (!visited.insert(cell).second) {
            return;
        }

        const auto surfaceIndexIt = surfaceIndexByPtr.find(tri.surface.get());
        if (surfaceIndexIt == surfaceIndexByPtr.end()) {
            return;
        }
        const int surfaceIndex = surfaceIndexIt->second;
        const auto& candidate = surfaces[static_cast<size_t>(surfaceIndex)];
        for (const auto& quadHit : rayBilinearQuadIntersections(src, rayDir, segmentLength, quad)) {
            ProjectionHit hit;
            hit.surface = tri.surface;
            hit.surfaceIndex = surfaceIndex;
            hit.surfaceName = candidate.name;
            hit.world = quadHit.world;
            hit.atlasU = static_cast<double>(col0) + quadHit.u * static_cast<double>(col1 - col0);
            hit.atlasV = static_cast<double>(row0) + quadHit.v * static_cast<double>(row1 - row0);
            hit.distance = distance(quadHit.world, linePoint);
            hits.push_back(hit);
        }
    });

    std::sort(hits.begin(), hits.end(), [](const ProjectionHit& a, const ProjectionHit& b) {
        return a.distance < b.distance;
    });
    return hits;
}

std::vector<ProjectionHit> projectPointToSurfacesAdaptive(
    const cv::Vec3d& linePoint,
    const cv::Vec3d& normal,
    const std::vector<SurfaceCandidate>& surfaces,
    const SurfacePatchIndex& index,
    double initialRayHalfLength)
{
    const double maxRayHalfLength = boundedRayHalfLength(
        linePoint, normal, surfaces, initialRayHalfLength);

    double rayHalfLength = std::isfinite(initialRayHalfLength) && initialRayHalfLength > 0.0
        ? initialRayHalfLength
        : 1.0;
    while (true) {
        rayHalfLength = std::min(rayHalfLength, maxRayHalfLength);
        auto hits = projectPointToSurfaces(
            linePoint, normal, surfaces, index, rayHalfLength);
        if (!hits.empty() || rayHalfLength >= maxRayHalfLength - kEpsilon) {
            return hits;
        }
        rayHalfLength = std::min(rayHalfLength * 2.0, maxRayHalfLength);
    }
}

AtlasAnchor anchorFromHit(int sourceIndex, const ProjectionHit& hit)
{
    AtlasAnchor anchor;
    anchor.sourceIndex = sourceIndex;
    anchor.world = hit.world;
    anchor.atlasU = hit.atlasU;
    anchor.atlasV = hit.atlasV;
    anchor.distance = hit.distance;
    return anchor;
}

struct ContinuationRejectDebug {
    int hitCount = 0;
    int candidateCount = 0;
    double lineStep = 0.0;
    double mismatchRatio = 0.0;
    double atlasNominalStepU = 1.0;
    double atlasNominalStepV = 1.0;
    double previousAtlasU = 0.0;
    double previousAtlasV = 0.0;
    int previousWinding = 0;
    double bestRejectedGridStep = std::numeric_limits<double>::infinity();
    double bestRejectedScaledAtlasStep = std::numeric_limits<double>::infinity();
    double bestRejectedRatio = std::numeric_limits<double>::infinity();
    double bestRejectedAtlasU = 0.0;
    double bestRejectedAtlasV = 0.0;
    double bestRejectedHitU = 0.0;
    double bestRejectedHitV = 0.0;
    double bestRejectedDistance = 0.0;
    int bestRejectedWinding = 0;
    std::string reason;
};

std::string continuationRejectDebugString(int sourceIndex,
                                          const ContinuationRejectDebug& debug)
{
    std::ostringstream out;
    out << "line_point[" << sourceIndex << "] continuation_rejection"
        << " reason=" << (debug.reason.empty() ? "unknown" : debug.reason)
        << " hits=" << debug.hitCount
        << " candidates=" << debug.candidateCount
        << " line_step=" << debug.lineStep
        << " atlas_nominal_step=(" << debug.atlasNominalStepU << ", "
        << debug.atlasNominalStepV << ")"
        << " threshold=" << debug.mismatchRatio
        << " prev_uv=(" << debug.previousAtlasU << ", " << debug.previousAtlasV << ")"
        << " prev_winding=" << debug.previousWinding;
    if (std::isfinite(debug.bestRejectedGridStep)) {
        out << " best_rejected_grid_step=" << debug.bestRejectedGridStep
            << " best_rejected_scaled_atlas_step=" << debug.bestRejectedScaledAtlasStep
            << " best_rejected_ratio=" << debug.bestRejectedRatio
            << " best_rejected_uv=(" << debug.bestRejectedAtlasU << ", "
            << debug.bestRejectedAtlasV << ")"
            << " raw_hit_uv=(" << debug.bestRejectedHitU << ", "
            << debug.bestRejectedHitV << ")"
            << " winding=" << debug.bestRejectedWinding
            << " projection_distance=" << debug.bestRejectedDistance;
    }
    return out.str();
}

std::optional<AtlasAnchor> chooseContinuationHit(int sourceIndex,
                                                 const std::vector<ProjectionHit>& hits,
                                                 const AtlasAnchor& previous,
                                                 const cv::Vec3d& previousLinePoint,
                                                 const cv::Vec3d& linePoint,
                                                 int periodColumns,
                                                 const cv::Vec2d& atlasNominalStep,
                                                 double mismatchRatio,
                                                 ContinuationRejectDebug* rejectDebug = nullptr)
{
    if (rejectDebug) {
        *rejectDebug = {};
        rejectDebug->hitCount = static_cast<int>(hits.size());
        rejectDebug->mismatchRatio = mismatchRatio;
        rejectDebug->atlasNominalStepU = atlasNominalStep[0];
        rejectDebug->atlasNominalStepV = atlasNominalStep[1];
        rejectDebug->previousAtlasU = previous.atlasU;
        rejectDebug->previousAtlasV = previous.atlasV;
        rejectDebug->previousWinding = periodColumns > 0
            ? static_cast<int>(std::floor(previous.atlasU / periodColumns))
            : 0;
    }
    if (hits.empty() || periodColumns <= 0) {
        if (rejectDebug) {
            rejectDebug->reason = hits.empty() ? "no_hits" : "invalid_period_columns";
        }
        return std::nullopt;
    }
    const double lineStep = distance(previousLinePoint, linePoint);
    if (rejectDebug) {
        rejectDebug->lineStep = lineStep;
    }
    double bestScore = std::numeric_limits<double>::infinity();
    std::optional<AtlasAnchor> best;
    double bestGridStep = std::numeric_limits<double>::infinity();
    double bestScaledAtlasStep = std::numeric_limits<double>::infinity();
    double bestRatio = 0.0;
    double bestHitU = 0.0;
    double bestHitV = 0.0;
    double bestDistance = 0.0;
    int bestWinding = 0;
    const int prevWinding = static_cast<int>(
        std::floor(previous.atlasU / static_cast<double>(periodColumns)));
    for (const auto& hit : hits) {
        for (int winding = prevWinding - 1; winding <= prevWinding + 1; ++winding) {
            if (rejectDebug) {
                ++rejectDebug->candidateCount;
            }
            AtlasAnchor candidate = anchorFromHit(sourceIndex, hit);
            candidate.atlasU = normalizeAtlasU(hit.atlasU, periodColumns) +
                               static_cast<double>(winding * periodColumns);
            const double du = candidate.atlasU - previous.atlasU;
            const double dv = candidate.atlasV - previous.atlasV;
            const double gridStep = std::sqrt(du * du + dv * dv);
            const double scaledDu = du * atlasNominalStep[0];
            const double scaledDv = dv * atlasNominalStep[1];
            const double scaledAtlasStep = std::sqrt(scaledDu * scaledDu + scaledDv * scaledDv);
            double ratio = 0.0;
            if (lineStep > kEpsilon) {
                ratio = scaledAtlasStep / lineStep;
            }
            if (gridStep < bestScore) {
                bestScore = gridStep;
                best = candidate;
                bestGridStep = gridStep;
                bestScaledAtlasStep = scaledAtlasStep;
                bestRatio = ratio;
                bestHitU = hit.atlasU;
                bestHitV = hit.atlasV;
                bestDistance = hit.distance;
                bestWinding = winding;
            }
        }
    }
    if (best && lineStep > kEpsilon && bestScaledAtlasStep > lineStep * mismatchRatio) {
        if (rejectDebug) {
            rejectDebug->reason = "step_mismatch";
            rejectDebug->bestRejectedGridStep = bestGridStep;
            rejectDebug->bestRejectedScaledAtlasStep = bestScaledAtlasStep;
            rejectDebug->bestRejectedRatio = bestRatio;
            rejectDebug->bestRejectedAtlasU = best->atlasU;
            rejectDebug->bestRejectedAtlasV = best->atlasV;
            rejectDebug->bestRejectedHitU = bestHitU;
            rejectDebug->bestRejectedHitV = bestHitV;
            rejectDebug->bestRejectedDistance = bestDistance;
            rejectDebug->bestRejectedWinding = bestWinding;
        }
        return std::nullopt;
    }
    if (!best && rejectDebug && rejectDebug->reason.empty()) {
        rejectDebug->reason = "no_acceptable_candidate";
    }
    return best;
}

} // namespace

void Atlas::save(const fs::path& atlasDir) const
{
    nlohmann::json metadataJson = {
        {"type", metadata.type},
        {"version", kAtlasMetadataVersion},
        {"name", metadata.name},
        {"base_mesh_path", metadata.baseMeshPath.generic_string()},
        {"source_base_mesh_path", metadata.sourceBaseMeshPath.generic_string()},
        {"zero_winding_column", metadata.zeroWindingColumn},
        {"seed_line_index", metadata.seedLineIndex},
        {"seed_atlas", nlohmann::json::array({metadata.seedAtlasU, metadata.seedAtlasV})},
    };
    writeJsonFile(atlasDir / "metadata.json", metadataJson);

    nlohmann::json linksJson;
    linksJson["version"] = 1;
    linksJson["links"] = nlohmann::json::array();
    for (const auto& link : links) {
        linksJson["links"].push_back(linkJson(link));
    }
    writeJsonFile(atlasDir / "links.json", linksJson);

    for (const auto& fiber : fibers) {
        nlohmann::json root;
        root["type"] = "vc3d_atlas_fiber_mapping";
        root["version"] = kFiberMappingVersion;
        root["fiber_path"] = fiber.fiberPath.generic_string();
        root["line_anchors"] = nlohmann::json::array();
        for (const auto& anchor : fiber.lineAnchors) {
            root["line_anchors"].push_back(anchorJson(anchor));
        }
        root["control_anchors"] = nlohmann::json::array();
        for (const auto& anchor : fiber.controlAnchors) {
            root["control_anchors"].push_back(anchorJson(anchor));
        }
        const std::string stem = fiber.fiberPath.stem().empty()
            ? std::string("fiber")
            : fiber.fiberPath.stem().string();
        writeJsonFile(atlasDir / "mappings" / "fibers" / (stem + ".json"), root);
    }
}

Atlas Atlas::load(const fs::path& atlasDir)
{
    Atlas atlas;
    const auto metadata = readJsonFile(atlasDir / "metadata.json");
    atlas.metadata.type = metadata.value("type", std::string{});
    atlas.metadata.version = metadata.value("version", 0);
    if (atlas.metadata.type != "vc3d_atlas") {
        throw std::runtime_error("unsupported atlas metadata in " + atlasDir.string());
    }
    if (atlas.metadata.version != kAtlasMetadataVersion) {
        throw std::runtime_error(
            "atlas metadata version " + std::to_string(atlas.metadata.version) +
            " is obsolete; rebuild required for " + atlasDir.string());
    }
    if (metadata.contains("idx_rotation_columns") || !metadata.contains("zero_winding_column")) {
        throw std::runtime_error("unsupported atlas metadata in " + atlasDir.string());
    }
    atlas.metadata.name = metadata.value("name", atlasDir.filename().string());
    atlas.metadata.baseMeshPath = metadata.value("base_mesh_path", std::string{});
    atlas.metadata.sourceBaseMeshPath = metadata.value("source_base_mesh_path", std::string{});
    atlas.metadata.zeroWindingColumn = metadata.at("zero_winding_column").get<int>();
    atlas.metadata.seedLineIndex = metadata.value("seed_line_index", 0);
    if (metadata.contains("seed_atlas") && metadata["seed_atlas"].is_array() &&
        metadata["seed_atlas"].size() == 2) {
        atlas.metadata.seedAtlasU = metadata["seed_atlas"][0].get<double>();
        atlas.metadata.seedAtlasV = metadata["seed_atlas"][1].get<double>();
    }

    const fs::path linksPath = atlasDir / "links.json";
    if (fs::exists(linksPath)) {
        const auto linksJson = readJsonFile(linksPath);
        if (linksJson.contains("links") && linksJson["links"].is_array()) {
            for (const auto& link : linksJson["links"]) {
                if (link.is_string()) {
                    continue;
                }
                atlas.links.push_back(linkFromJson(link));
            }
        }
    }

    const fs::path mappingsDir = atlasDir / "mappings" / "fibers";
    if (fs::is_directory(mappingsDir)) {
        for (const auto& mappingPath : sortedAtlasFiberMappingFiles(atlasDir)) {
            const auto root = readJsonFile(mappingPath);
            if (!root.contains("version")) {
                throw std::runtime_error("atlas fiber mapping " + mappingPath.string() +
                                         " is missing version; rebuild required");
            }
            const int mappingVersion = root.at("version").get<int>();
            if (mappingVersion != kFiberMappingVersion) {
                throw std::runtime_error(
                    "atlas fiber mapping " + mappingPath.string() +
                    " has obsolete version " + std::to_string(mappingVersion) +
                    "; rebuild required");
            }
            FiberMapping mapping;
            mapping.fiberPath = root.value("fiber_path", std::string{});
            mapping.windingOffset = 0;
            for (const auto& anchor : root.value("line_anchors", nlohmann::json::array())) {
                mapping.lineAnchors.push_back(anchorFromJson(anchor));
            }
            for (const auto& anchor : root.value("control_anchors", nlohmann::json::array())) {
                mapping.controlAnchors.push_back(anchorFromJson(anchor));
            }
            atlas.fibers.push_back(std::move(mapping));
        }
    }
    return atlas;
}

bool atlasLoadErrorRequiresRebuild(const std::exception& ex)
{
    const std::string message = ex.what();
    return message.find("rebuild required") != std::string::npos ||
           message.find("obsolete") != std::string::npos;
}

Atlas rebuildAtlasFromSourceFibers(const fs::path& atlasDir,
                                   const fs::path& volpkgRoot,
                                   const vc::lasagna::NormalSampler& normalSampler,
                                   const LineMappingOptions& options)
{
    const Atlas legacy = loadAtlasContextForRebuild(atlasDir);
    if (legacy.metadata.baseMeshPath.empty()) {
        throw std::runtime_error("cannot rebuild atlas without base_mesh_path");
    }
    QuadSurface loadedBase(atlasDir / legacy.metadata.baseMeshPath);
    const auto* points = loadedBase.rawPointsPtr();
    if (!points || points->empty()) {
        throw std::runtime_error("cannot rebuild atlas: base mesh has no valid grid");
    }
    auto baseSurface = std::make_shared<QuadSurface>(*points, loadedBase.scale());
    SurfacePatchIndex baseIndex;
    baseIndex.rebuild({baseSurface});
    return rebuildAtlasFromSourceFibers(
        atlasDir, volpkgRoot, *baseSurface, baseIndex, normalSampler, options);
}

Atlas rebuildAtlasFromSourceFibers(const fs::path& atlasDir,
                                   const fs::path& volpkgRootIn,
                                   const QuadSurface& baseSurface,
                                   SurfacePatchIndex& baseIndex,
                                   const vc::lasagna::NormalSampler& normalSampler,
                                   const LineMappingOptions& options)
{
    const fs::path volpkgRoot = volpkgRootIn.empty()
        ? inferVolpkgRootFromAtlasDir(atlasDir)
        : volpkgRootIn;
    if (volpkgRoot.empty()) {
        throw std::runtime_error("cannot rebuild atlas without a volume package root");
    }

    const Atlas legacy = loadAtlasContextForRebuild(atlasDir);
    Atlas rebuilt;
    rebuilt.metadata = legacy.metadata;
    rebuilt.metadata.version = kAtlasMetadataVersion;
    rebuilt.links = legacy.links;
    rebuilt.fibers.reserve(legacy.fibers.size());

    for (const auto& oldMapping : legacy.fibers) {
        if (oldMapping.fiberPath.empty()) {
            throw std::runtime_error("cannot rebuild atlas mapping with missing fiber path");
        }
        const fs::path fiberPath = resolveAtlasRelativePath(
            atlasDir, volpkgRoot, oldMapping.fiberPath);
        const FiberInput input = loadSourceFiberInput(fiberPath, oldMapping.fiberPath);
        rebuilt.fibers.push_back(
            mapFiberToBaseSurface(input, baseSurface, baseIndex, normalSampler, options));
    }

    auto refreshEndpoint = [&rebuilt](AtlasLinkEndpoint& endpoint) {
        const std::string endpointKey = atlasFiberPathKey(endpoint.fiberPath);
        const auto mappingIt = std::find_if(
            rebuilt.fibers.begin(),
            rebuilt.fibers.end(),
            [&endpointKey](const FiberMapping& mapping) {
                return atlasFiberPathKey(mapping.fiberPath) == endpointKey;
            });
        if (mappingIt == rebuilt.fibers.end()) {
            throw std::runtime_error(
                "cannot rebuild atlas link endpoint for missing fiber " +
                endpoint.fiberPath.generic_string());
        }
        const auto anchorIt = std::find_if(
            mappingIt->lineAnchors.begin(),
            mappingIt->lineAnchors.end(),
            [&endpoint](const AtlasAnchor& anchor) {
                return anchor.sourceIndex == endpoint.sourceIndex;
            });
        if (anchorIt == mappingIt->lineAnchors.end()) {
            throw std::runtime_error(
                "cannot rebuild atlas link endpoint " +
                endpoint.fiberPath.generic_string() +
                " source_index " + std::to_string(endpoint.sourceIndex) +
                ": mapped line anchor not found");
        }
        endpoint.atlasU = anchorIt->atlasU;
        endpoint.atlasV = anchorIt->atlasV;
    };

    for (auto& link : rebuilt.links) {
        refreshEndpoint(link.first);
        refreshEndpoint(link.second);
    }
    layoutAtlasObjects(rebuilt, atlasHorizontalPeriodColumns(baseSurface));
    rebuilt.save(atlasDir);
    return rebuilt;
}

std::string sanitizeAtlasName(std::string name)
{
    for (char& ch : name) {
        const auto c = static_cast<unsigned char>(ch);
        if (!std::isalnum(c) && ch != '_' && ch != '-') {
            ch = '_';
        }
    }
    while (!name.empty() && name.front() == '_') name.erase(name.begin());
    while (!name.empty() && name.back() == '_') name.pop_back();
    return name.empty() ? "atlas" : name;
}

std::string atlasFiberPathKey(const fs::path& path)
{
    return path.lexically_normal().generic_string();
}

std::vector<std::string> atlasMappedFiberPathKeys(const Atlas& atlas)
{
    std::vector<std::string> keys;
    keys.reserve(atlas.fibers.size());
    for (const auto& mapping : atlas.fibers) {
        if (!mapping.fiberPath.empty()) {
            keys.push_back(atlasFiberPathKey(mapping.fiberPath));
        }
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    return keys;
}

uint64_t FiberRuntimeIdentityMap::idForPath(const fs::path& path) const
{
    const auto it = idByPathKey.find(atlasFiberPathKey(path));
    if (it == idByPathKey.end()) {
        throw std::out_of_range("fiber path is not in the runtime identity map: " +
                                path.generic_string());
    }
    return it->second;
}

fs::path FiberRuntimeIdentityMap::pathForId(uint64_t id) const
{
    const auto it = pathById.find(id);
    if (it == pathById.end()) {
        throw std::out_of_range("fiber runtime id is not in the identity map: " +
                                std::to_string(id));
    }
    return it->second;
}

FiberRuntimeIdentityMap makeFiberRuntimeIdentityMap(
    const std::vector<fs::path>& orderedCanonicalFiberPaths)
{
    FiberRuntimeIdentityMap map;
    map.canonicalPaths.reserve(orderedCanonicalFiberPaths.size());
    uint64_t nextId = 1;
    for (const auto& path : orderedCanonicalFiberPaths) {
        const std::string key = atlasFiberPathKey(path);
        if (key.empty() || map.idByPathKey.find(key) != map.idByPathKey.end()) {
            continue;
        }
        const uint64_t id = nextId++;
        const fs::path canonicalPath(key);
        map.canonicalPaths.push_back(canonicalPath);
        map.idByPathKey.emplace(key, id);
        map.pathById.emplace(id, canonicalPath);
    }
    return map;
}

AtlasFiberSearchSets atlasFiberSearchSets(const Atlas& atlas,
                                          const FiberRuntimeIdentityMap& runtimeIds)
{
    const std::vector<std::string> atlasKeys = atlasMappedFiberPathKeys(atlas);
    AtlasFiberSearchSets sets;
    sets.sourceFiberIds.reserve(atlasKeys.size());
    sets.sourceFiberPaths.reserve(atlasKeys.size());
    sets.targetFiberIds.reserve(runtimeIds.canonicalPaths.size());
    sets.targetFiberPaths.reserve(runtimeIds.canonicalPaths.size());
    for (const auto& path : runtimeIds.canonicalPaths) {
        const std::string key = atlasFiberPathKey(path);
        const uint64_t id = runtimeIds.idForPath(path);
        if (std::binary_search(atlasKeys.begin(), atlasKeys.end(), key)) {
            sets.sourceFiberIds.push_back(id);
            sets.sourceFiberPaths.push_back(path);
        } else {
            sets.targetFiberIds.push_back(id);
            sets.targetFiberPaths.push_back(path);
        }
    }
    return sets;
}

std::vector<AtlasDirectoryInfo> discoverAtlasDirectories(const fs::path& volpkgRoot)
{
    std::vector<AtlasDirectoryInfo> out;
    const fs::path atlasRoot = volpkgRoot / "atlases";
    if (volpkgRoot.empty() || !fs::is_directory(atlasRoot)) {
        return out;
    }

    std::vector<fs::path> atlasDirs;
    for (const auto& entry : fs::directory_iterator(atlasRoot)) {
        if (entry.is_directory() && fs::exists(entry.path() / "metadata.json")) {
            atlasDirs.push_back(entry.path());
        }
    }
    std::sort(atlasDirs.begin(), atlasDirs.end());

    out.reserve(atlasDirs.size());
    for (const auto& atlasDir : atlasDirs) {
        try {
            const auto metadata = readJsonFile(atlasDir / "metadata.json");
            if (metadata.value("type", std::string{}) != "vc3d_atlas") {
                continue;
            }
            std::string name = metadata.value("name", atlasDir.filename().string());
            if (name.empty()) {
                name = atlasDir.filename().string();
            }
            out.push_back({atlasDir, name});
        } catch (...) {
            continue;
        }
    }
    return out;
}

LasagnaAtlasExport loadLasagnaAtlasExport(const fs::path& atlasDir,
                                          const fs::path& volpkgRootIn)
{
    if (atlasDir.empty() || !fs::is_directory(atlasDir)) {
        throw std::runtime_error("Atlas directory not found: " + atlasDir.string());
    }
    if (!fs::is_regular_file(atlasDir / "metadata.json")) {
        throw std::runtime_error("Atlas metadata.json not found: " + atlasDir.string());
    }

    LasagnaAtlasExport exportData;
    exportData.atlasDir = atlasDir;
    exportData.volpkgRoot = volpkgRootIn.empty()
        ? inferVolpkgRootFromAtlasDir(atlasDir)
        : volpkgRootIn;
    exportData.atlas = Atlas::load(atlasDir);
    exportData.baseRelativePath = exportData.atlas.metadata.baseMeshPath;
    if (exportData.baseRelativePath.empty()) {
        throw std::runtime_error("Atlas metadata is missing base_mesh_path.");
    }
    exportData.basePath = (atlasDir / exportData.baseRelativePath).lexically_normal();
    if (!fs::is_directory(exportData.basePath)) {
        throw std::runtime_error("Atlas base mesh does not exist: " + exportData.basePath.string());
    }
    try {
        const QuadSurface base(exportData.basePath);
        const cv::Mat_<cv::Vec3f>* points = base.rawPointsPtr();
        if (!points || points->rows < 2 || points->cols < 2) {
            throw std::runtime_error("Atlas base mesh is too small: " +
                                     exportData.basePath.string());
        }
        double maxDelta = 0.0;
        for (int row = 0; row < points->rows; ++row) {
            const cv::Vec3f first = (*points)(row, 0);
            const cv::Vec3f last = (*points)(row, points->cols - 1);
            if (!finitePoint(first) || !finitePoint(last)) {
                throw std::runtime_error(
                    "Atlas base mesh contains invalid wrap endpoints: " +
                    exportData.basePath.string());
            }
            for (int c = 0; c < 3; ++c) {
                maxDelta = std::max(
                    maxDelta,
                    std::abs(static_cast<double>(first[c] - last[c])));
            }
        }
        if (maxDelta > 1.0e-4) {
            std::ostringstream message;
            message << "Atlas base mesh is not explicitly wrapped; first/last column max delta is "
                    << maxDelta << '.';
            throw std::runtime_error(message.str());
        }
        const int periodColumns = atlasHorizontalPeriodColumns(base);
        layoutAtlasObjects(exportData.atlas, periodColumns);
    } catch (const std::exception& ex) {
        throw std::runtime_error("Cannot load atlas base mesh " +
                                 exportData.basePath.string() + ": " + ex.what());
    }

    const fs::path mappingsDir = atlasDir / "mappings" / "fibers";
    if (!fs::is_directory(mappingsDir)) {
        throw std::runtime_error("Atlas has no fiber mappings directory: " +
                                 mappingsDir.string());
    }
    const std::vector<fs::path> mappingFiles = sortedAtlasFiberMappingFiles(atlasDir);
    if (mappingFiles.empty()) {
        throw std::runtime_error("Atlas has no mapped fiber JSON files.");
    }
    if (mappingFiles.size() != exportData.atlas.fibers.size()) {
        throw std::runtime_error("Atlas fiber mapping count does not match loaded atlas state.");
    }

    exportData.objects.reserve(mappingFiles.size());
    for (size_t i = 0; i < mappingFiles.size(); ++i) {
        const FiberMapping& mapping = exportData.atlas.fibers[i];
        if (mapping.fiberPath.empty()) {
            throw std::runtime_error("Atlas mapping " + mappingFiles[i].string() +
                                     " references missing fiber path: ");
        }
        const fs::path fiberPath = resolveAtlasRelativePath(
            atlasDir, exportData.volpkgRoot, mapping.fiberPath);
        if (!fs::is_regular_file(fiberPath)) {
            throw std::runtime_error("Atlas mapping " + mappingFiles[i].string() +
                                     " references missing fiber path: " +
                                     mapping.fiberPath.generic_string());
        }

        LasagnaAtlasObject object;
        object.id = mapping.fiberPath.generic_string();
        object.fiberPath = fiberPath;
        object.mappingPath = mappingFiles[i];
        object.fiberRelativePath = mapping.fiberPath;
        object.mappingRelativePath = fs::relative(mappingFiles[i], atlasDir).lexically_normal();
        object.windingOffset = mapping.windingOffset;
        exportData.objects.push_back(std::move(object));
    }

    const std::string atlasName = exportData.atlas.metadata.name.empty()
        ? atlasDir.filename().string()
        : exportData.atlas.metadata.name;
    nlohmann::json lineObjects = nlohmann::json::array();
    nlohmann::json maps = nlohmann::json::array();
    std::unordered_set<std::string> lineIds;
    for (const auto& object : exportData.objects) {
        if (lineIds.insert(object.id).second) {
            lineObjects.push_back({
                {"id", object.id},
                {"fiber_path", object.fiberRelativePath.generic_string()},
            });
        }
        maps.push_back({
            {"object_type", "line"},
            {"object_id", object.id},
            {"fiber_path", object.fiberRelativePath.generic_string()},
            {"mapping_path", object.mappingRelativePath.generic_string()},
            {"winding_offset", object.windingOffset},
        });
    }

    exportData.compactJson = {
        {"type", "lasagna_atlas"},
        {"version", 1},
        {"name", atlasName},
        {"base", {
            {"path", exportData.baseRelativePath.generic_string()},
        }},
        {"metadata", {
            {"zero_winding_column", exportData.atlas.metadata.zeroWindingColumn},
        }},
        {"objects", {
            {"line", lineObjects},
        }},
        {"maps", maps},
    };
    return exportData;
}

fs::path uniqueAtlasDirectory(const fs::path& volpkgRoot, const std::string& baseName)
{
    const std::string clean = sanitizeAtlasName(baseName);
    const fs::path root = volpkgRoot / "atlases";
    fs::path candidate = root / clean;
    for (int suffix = 2; fs::exists(candidate); ++suffix) {
        candidate = root / (clean + "_" + std::to_string(suffix));
    }
    return candidate;
}

fs::path initShellDirectoryFromManifest(const vc::lasagna::LasagnaDatasetManifest& manifest)
{
    if (!manifest.initShellDir.has_value()) {
        throw std::runtime_error("Lasagna manifest is missing init_shell_dir");
    }
    if (!fs::is_directory(*manifest.initShellDir)) {
        throw std::runtime_error("Lasagna init_shell_dir does not exist or is not a directory: " +
                                 manifest.initShellDir->string());
    }
    return *manifest.initShellDir;
}

std::vector<SurfaceCandidate> loadInitShellCandidates(const fs::path& initShellDir)
{
    if (!fs::is_directory(initShellDir)) {
        throw std::runtime_error("Lasagna init_shell_dir does not exist or is not a directory: " +
                                 initShellDir.string());
    }

    std::vector<fs::path> shellDirs;
    for (const auto& entry : fs::directory_iterator(initShellDir)) {
        if (!entry.is_directory()) {
            continue;
        }
        const fs::path path = entry.path();
        const std::string filename = path.filename().string();
        if (filename.rfind("shell_", 0) != 0 || path.extension() != ".tifxyz") {
            continue;
        }
        shellDirs.push_back(path);
    }
    std::sort(shellDirs.begin(), shellDirs.end());

    std::vector<SurfaceCandidate> candidates;
    candidates.reserve(shellDirs.size());
    for (const auto& shellDir : shellDirs) {
        auto surface = std::make_shared<QuadSurface>(shellDir);
        std::string name = shellDir.stem().string();
        candidates.push_back({std::move(name), shellDir, std::move(surface)});
    }
    if (candidates.empty()) {
        throw std::runtime_error("Lasagna init_shell_dir contains no shell_*.tifxyz directories: " +
                                 initShellDir.string());
    }
    return candidates;
}

std::vector<ProjectionHit> projectPointAlongNormalToSurfaces(
    const cv::Vec3d& linePoint,
    const cv::Vec3d& normal,
    const std::vector<SurfaceCandidate>& surfaces,
    const SurfacePatchIndex& index,
    double rayHalfLength)
{
    return projectPointToSurfaces(linePoint, normal, surfaces, index, rayHalfLength);
}

BaseSelection selectBaseSurfaceBySeedRay(const FiberInput& fiber,
                                         const std::vector<SurfaceCandidate>& surfaces,
                                         const SurfacePatchIndex& index,
                                         const vc::lasagna::NormalSampler& normalSampler,
                                         const LineMappingOptions& options)
{
    if (surfaces.empty()) {
        throw std::runtime_error("no candidate shell surfaces are available");
    }
    const int seedIndex = seedLineIndexForFiber(fiber);
    const cv::Vec3d seedPoint = fiber.linePoints.at(seedIndex);
    atlasDebug("fiber line_points=" + std::to_string(fiber.linePoints.size()) +
               " control_points=" + std::to_string(fiber.controlPoints.size()) +
               " seed_index=" + std::to_string(seedIndex));
    const auto normalSample = normalSampler.sampleNormal(seedPoint);
    if (!normalSample.valid || !validNormal(normalSample.normal)) {
        std::ostringstream message;
        message << "No valid normal at atlas seed point"
                << " seed_index=" << seedIndex
                << " seed=" << vecString(seedPoint);
        throw std::runtime_error(message.str());
    }

    const auto hits = projectPointToSurfacesAdaptive(seedPoint,
                                                    normalSample.normal,
                                                    surfaces,
                                                    index,
                                                    options.rayHalfLength);
    if (atlasDebugEnabled()) {
        std::ostringstream out;
        out << "seed=" << vecString(seedPoint)
            << " normal=" << vecString(normalSample.normal)
            << " ray_hits=" << hits.size();
        for (const auto& hit : hits) {
            out << " [" << hit.surfaceName
                << " u=" << hit.atlasU
                << " v=" << hit.atlasV
                << " d=" << hit.distance << ']';
        }
        atlasDebug(out.str());
    }
    if (hits.empty()) {
        std::ostringstream message;
        message << "Atlas seed ray did not intersect any shell"
                << " seed_index=" << seedIndex
                << " seed=" << vecString(seedPoint)
                << " normal=" << vecString(normalSample.normal);
        throw std::runtime_error(message.str());
    }

    const auto& best = hits.front();
    BaseSelection selection;
    selection.surfaceIndex = best.surfaceIndex;
    selection.surfaceName = best.surfaceName;
    selection.seedPoint = seedPoint;
    selection.seedLineIndex = seedIndex;
    selection.world = best.world;
    selection.atlasU = best.atlasU;
    selection.atlasV = best.atlasV;
    selection.distance = best.distance;
    atlasDebug("selected_shell=" + selection.surfaceName +
               " seed_atlas=(" + std::to_string(selection.atlasU) + ", " +
               std::to_string(selection.atlasV) + ")");
    return selection;
}

int computeZeroWindingColumn(const QuadSurface& surface)
{
    const auto* points = surface.rawPointsPtr();
    if (!points || points->cols <= 0) {
        return 0;
    }
    const int periodColumns = atlasHorizontalPeriodColumns(surface);

    int bestCol = 0;
    double bestAverageY = std::numeric_limits<double>::infinity();
    for (int col = 0; col < periodColumns; ++col) {
        double sumY = 0.0;
        int count = 0;
        for (int row = 0; row < points->rows; ++row) {
            const cv::Vec3f p = (*points)(row, col);
            if (p[0] == -1.0f || !finitePoint(p)) {
                continue;
            }
            sumY += p[1];
            ++count;
        }
        if (count == 0) {
            continue;
        }
        const double averageY = sumY / static_cast<double>(count);
        if (averageY < bestAverageY) {
            bestAverageY = averageY;
            bestCol = col;
        }
    }
    return bestCol;
}

void saveAtlasBaseMeshCopy(const QuadSurface& surface,
                           const fs::path& targetDir)
{
    const auto* points = surface.rawPointsPtr();
    if (!points) {
        throw std::runtime_error("base surface has no point grid");
    }
    QuadSurface copy(*points, surface.scale());
    auto& mutableSurface = const_cast<QuadSurface&>(surface);
    for (const auto& name : mutableSurface.channelNames()) {
        cv::Mat channel = mutableSurface.channel(name, SURF_CHANNEL_NORESIZE);
        if (channel.empty()) {
            continue;
        }
        copy.setChannel(name, channel.clone());
    }
    copy.save(targetDir, true);
}

AtlasCoveredSize mappedObjectCoveredAtlasSize(const Atlas& atlas,
                                              cv::Vec2f atlasScale,
                                              int periodColumns)
{
    if (!std::isfinite(atlasScale[0]) || !std::isfinite(atlasScale[1]) ||
        atlasScale[0] <= 0.0f || atlasScale[1] <= 0.0f) {
        throw std::runtime_error("atlas base mesh has invalid scale");
    }
    const double scaleX = static_cast<double>(atlasScale[0]);
    const double scaleY = static_cast<double>(atlasScale[1]);

    bool haveAnchor = false;
    double minU = std::numeric_limits<double>::infinity();
    double minV = std::numeric_limits<double>::infinity();
    double maxU = -std::numeric_limits<double>::infinity();
    double maxV = -std::numeric_limits<double>::infinity();

    auto includeAnchor = [&](const AtlasAnchor& anchor, const FiberMapping& fiber) {
        if (!std::isfinite(anchor.atlasU) || !std::isfinite(anchor.atlasV)) {
            return;
        }
        const double atlasU = actualAtlasU(anchor, fiber, periodColumns);
        haveAnchor = true;
        minU = std::min(minU, atlasU);
        minV = std::min(minV, anchor.atlasV);
        maxU = std::max(maxU, atlasU);
        maxV = std::max(maxV, anchor.atlasV);
    };

    for (const auto& fiber : atlas.fibers) {
        for (const auto& anchor : fiber.lineAnchors) {
            includeAnchor(anchor, fiber);
        }
    }

    if (!haveAnchor) {
        return {};
    }

    AtlasCoveredSize size;
    size.width = (maxU - minU) / scaleX;
    size.height = (maxV - minV) / scaleY;
    size.valid = true;
    return size;
}

int atlasHorizontalPeriodColumns(const QuadSurface& surface)
{
    const auto* points = surface.rawPointsPtr();
    if (!points || points->empty() || points->rows <= 0 || points->cols <= 0) {
        throw std::runtime_error("atlas init shell has no valid grid");
    }
    const int cols = points->cols;
    if (cols < 2) {
        throw std::runtime_error("atlas init shell must have at least two columns");
    }

    for (int row = 0; row < points->rows; ++row) {
        const cv::Vec3f first = (*points)(row, 0);
        const cv::Vec3f last = (*points)(row, cols - 1);
        if (!finitePoint(first) || !finitePoint(last)) {
            throw std::runtime_error(
                "atlas init shell is not explicitly wrapped: first and last columns differ");
        }
        const cv::Vec3d delta = toVec3d(first) - toVec3d(last);
        if (norm(delta) > 1.0e-5) {
            throw std::runtime_error(
                "atlas init shell is not explicitly wrapped: first and last columns differ");
        }
    }
    return cols - 1;
}

int atlasWindingForColumn(double atlasU, int periodColumns, int zeroWindingColumn)
{
    if (periodColumns <= 0 || !std::isfinite(atlasU)) {
        return 0;
    }
    const double period = static_cast<double>(periodColumns);
    return static_cast<int>(
        std::floor((atlasU - static_cast<double>(zeroWindingColumn)) / period));
}

double actualAtlasU(const AtlasAnchor& anchor,
                    const FiberMapping& fiber,
                    int periodColumns)
{
    if (periodColumns <= 0 || !std::isfinite(anchor.atlasU)) {
        return anchor.atlasU;
    }
    return anchor.atlasU + static_cast<double>(fiber.windingOffset * periodColumns);
}

std::optional<cv::Vec3d> atlasBasePointAt(double atlasU,
                                          double atlasV,
                                          const QuadSurface& baseSurface)
{
    const auto* points = baseSurface.rawPointsPtr();
    if (!points || points->empty() ||
        !std::isfinite(atlasU) || !std::isfinite(atlasV)) {
        return std::nullopt;
    }
    const int periodColumns = atlasHorizontalPeriodColumns(baseSurface);
    const double baseU = normalizeAtlasU(atlasU, periodColumns);
    const cv::Vec2d grid{baseU, atlasV};
    if (!loc_valid_xy(*points, grid)) {
        return std::nullopt;
    }
    const cv::Vec3f p = at_int(*points, cv::Vec2f(static_cast<float>(baseU),
                                                  static_cast<float>(atlasV)));
    if (!finitePoint(p)) {
        return std::nullopt;
    }
    return toVec3d(p);
}

std::optional<cv::Vec3d> atlasAnchorBasePoint(const AtlasAnchor& anchor,
                                              const FiberMapping& fiber,
                                              const QuadSurface& baseSurface)
{
    const int periodColumns = atlasHorizontalPeriodColumns(baseSurface);
    return atlasBasePointAt(actualAtlasU(anchor, fiber, periodColumns),
                            anchor.atlasV,
                            baseSurface);
}

AtlasDisplayRange atlasDisplayRange(const Atlas& atlas, int baseColumns)
{
    AtlasDisplayRange range;
    range.baseColumns = baseColumns;
    if (baseColumns <= 0) {
        range.unwrapCount = 0;
        return range;
    }

    bool haveAnchor = false;
    int minWinding = 0;
    int maxWinding = 0;
    auto includeAnchor = [&](const AtlasAnchor& anchor, const FiberMapping& fiber) {
        if (!std::isfinite(anchor.atlasU)) {
            return;
        }
        const double atlasU = actualAtlasU(anchor, fiber, baseColumns);
        const int winding = atlasWindingForColumn(
            atlasU, baseColumns, atlas.metadata.zeroWindingColumn);
        if (!haveAnchor) {
            minWinding = winding;
            maxWinding = winding;
            haveAnchor = true;
            return;
        }
        minWinding = std::min(minWinding, winding);
        maxWinding = std::max(maxWinding, winding);
    };

    for (const auto& fiber : atlas.fibers) {
        for (const auto& anchor : fiber.lineAnchors) {
            includeAnchor(anchor, fiber);
        }
    }

    if (!haveAnchor) {
        minWinding = atlasWindingForColumn(
            atlas.metadata.seedAtlasU, baseColumns, atlas.metadata.zeroWindingColumn);
        maxWinding = minWinding;
    }

    range.leftmostWinding = minWinding;
    range.rightmostWinding = maxWinding;
    range.unwrapCount = std::max(1, maxWinding - minWinding + 1);
    range.atlasUOffset = static_cast<double>(atlas.metadata.zeroWindingColumn) +
                         static_cast<double>(minWinding * baseColumns);
    range.hasMappedObjects = haveAnchor;
    return range;
}

void layoutAtlasObjects(Atlas& atlas, int periodColumns)
{
    if (periodColumns <= 0 || atlas.fibers.empty()) {
        return;
    }

    std::unordered_map<std::string, size_t> fiberIndexByPath;
    fiberIndexByPath.reserve(atlas.fibers.size());
    for (size_t i = 0; i < atlas.fibers.size(); ++i) {
        fiberIndexByPath.emplace(atlas.fibers[i].fiberPath.generic_string(), i);
        atlas.fibers[i].windingOffset = 0;
    }

    struct Edge {
        size_t to = 0;
        int delta = 0;
    };
    std::vector<std::vector<Edge>> graph(atlas.fibers.size());
    for (const auto& link : atlas.links) {
        const auto firstIt = fiberIndexByPath.find(link.first.fiberPath.generic_string());
        const auto secondIt = fiberIndexByPath.find(link.second.fiberPath.generic_string());
        if (firstIt == fiberIndexByPath.end() || secondIt == fiberIndexByPath.end()) {
            continue;
        }
        const int firstBaseWinding = atlasWindingForColumn(
            link.first.atlasU, periodColumns, atlas.metadata.zeroWindingColumn);
        const int secondBaseWinding = atlasWindingForColumn(
            link.second.atlasU, periodColumns, atlas.metadata.zeroWindingColumn);
        const int secondMinusFirst =
            link.desiredWindingDelta - (secondBaseWinding - firstBaseWinding);
        graph[firstIt->second].push_back({secondIt->second, secondMinusFirst});
        graph[secondIt->second].push_back({firstIt->second, -secondMinusFirst});
    }

    std::vector<bool> visited(atlas.fibers.size(), false);
    std::queue<size_t> pending;
    visited[0] = true;
    pending.push(0);
    while (!pending.empty()) {
        const size_t current = pending.front();
        pending.pop();
        for (const auto& edge : graph[current]) {
            const int candidateOffset = atlas.fibers[current].windingOffset + edge.delta;
            if (visited[edge.to]) {
                continue;
            }
            atlas.fibers[edge.to].windingOffset = candidateOffset;
            visited[edge.to] = true;
            pending.push(edge.to);
        }
    }
}

cv::Vec2f atlasGridToSurfaceCoords(double atlasU,
                                   double atlasV,
                                   const QuadSurface& displaySurface,
                                   double atlasUOffset)
{
    const auto* points = displaySurface.rawPointsPtr();
    const cv::Vec2f scale = displaySurface.scale();
    if (!points || points->empty() ||
        !std::isfinite(atlasU) || !std::isfinite(atlasV) ||
        !std::isfinite(atlasUOffset) ||
        !std::isfinite(scale[0]) || !std::isfinite(scale[1]) ||
        scale[0] == 0.0f || scale[1] == 0.0f) {
        return {std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN()};
    }
    return {
        static_cast<float>(((atlasU - atlasUOffset) - static_cast<double>(points->cols) / 2.0) /
                           static_cast<double>(scale[0])),
        static_cast<float>((atlasV - static_cast<double>(points->rows) / 2.0) /
                           static_cast<double>(scale[1])),
    };
}

std::shared_ptr<QuadSurface> repeatedAtlasDisplaySurface(const QuadSurface& baseSurface,
                                                        int unwrapCount,
                                                        int startColumn)
{
    const auto* points = baseSurface.rawPointsPtr();
    if (!points) {
        throw std::runtime_error("base surface has no point grid");
    }
    if (unwrapCount <= 0) {
        throw std::runtime_error("atlas display unwrap count must be positive");
    }

    const int rows = points->rows;
    const int cols = points->cols;
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("base surface has no valid grid");
    }

    const int periodColumns = atlasHorizontalPeriodColumns(baseSurface);
    const int outCols = periodColumns * unwrapCount;
    const int start = ((startColumn % periodColumns) + periodColumns) % periodColumns;

    cv::Mat_<cv::Vec3f> repeated(rows, outCols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < outCols; ++col) {
            repeated(row, col) = (*points)(row, (start + col) % periodColumns);
        }
    }

    auto out = std::make_shared<QuadSurface>(repeated, baseSurface.scale());
    auto& mutableSurface = const_cast<QuadSurface&>(baseSurface);
    for (const auto& name : mutableSurface.channelNames()) {
        cv::Mat channel = mutableSurface.channel(name);
        if (channel.empty() || channel.cols != cols || channel.rows != rows) {
            continue;
        }
        cv::Mat repeatedChannel(rows, outCols, channel.type());
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < outCols; ++col) {
                channel(cv::Rect((start + col) % periodColumns, row, 1, 1)).copyTo(
                    repeatedChannel(cv::Rect(col, row, 1, 1)));
            }
        }
        out->setChannel(name, repeatedChannel);
    }
    return out;
}

void validateFiberInputControlPoints(FiberInput& fiber)
{
    fiber.controlLineIndices.clear();
    fiber.controlLineIndices.reserve(fiber.controlPoints.size());

    for (size_t i = 0; i < fiber.linePoints.size(); ++i) {
        if (!finitePoint(fiber.linePoints[i])) {
            throw std::runtime_error("fiber line_points[" + std::to_string(i) +
                                     "] contains non-finite coordinates");
        }
    }

    int nextLineIndex = 0;
    const double maxDistanceSq = kControlPointMatchEpsilon * kControlPointMatchEpsilon;
    for (size_t i = 0; i < fiber.controlPoints.size(); ++i) {
        if (!finitePoint(fiber.controlPoints[i])) {
            throw std::runtime_error("fiber control_points[" + std::to_string(i) +
                                     "] contains non-finite coordinates");
        }
        int matchedLineIndex = -1;
        for (int j = nextLineIndex; j < static_cast<int>(fiber.linePoints.size()); ++j) {
            if (squaredDistance(fiber.controlPoints[i], fiber.linePoints[static_cast<size_t>(j)]) <=
                maxDistanceSq) {
                matchedLineIndex = j;
                break;
            }
        }
        if (matchedLineIndex < 0) {
            throw std::runtime_error(
                "fiber control_points[" + std::to_string(i) +
                "] is not an ordered subset of line_points; rebuild or repair the fiber JSON");
        }
        fiber.controlLineIndices.push_back(matchedLineIndex);
        nextLineIndex = matchedLineIndex + 1;
    }
}

FiberMapping mapFiberToBaseSurface(const FiberInput& fiber,
                                   const QuadSurface& baseSurface,
                                   SurfacePatchIndex& baseIndex,
                                   const vc::lasagna::NormalSampler& normalSampler,
                                   const LineMappingOptions& options)
{
    FiberInput validatedFiber = fiber;
    validateFiberInputControlPoints(validatedFiber);

    if (validatedFiber.linePoints.empty()) {
        throw std::runtime_error("fiber has no line points");
    }
    const auto* points = baseSurface.rawPointsPtr();
    if (!points || points->cols <= 0) {
        throw std::runtime_error("base surface has no valid grid");
    }
    const int periodColumns = atlasHorizontalPeriodColumns(baseSurface);
    const cv::Vec2f baseScale = baseSurface.scale();
    const cv::Vec2d atlasNominalStep{
        std::isfinite(baseScale[0]) && baseScale[0] > 0.0f ? 1.0 / static_cast<double>(baseScale[0]) : 1.0,
        std::isfinite(baseScale[1]) && baseScale[1] > 0.0f ? 1.0 / static_cast<double>(baseScale[1]) : 1.0,
    };

    SurfacePatchIndex::SurfacePtr baseSurfacePtr(const_cast<QuadSurface*>(&baseSurface), [](QuadSurface*) {});
    const std::vector<SurfaceCandidate> baseCandidates = {{
        "base",
        {},
        baseSurfacePtr,
    }};

    const int seedIndex = seedLineIndexForFiber(validatedFiber);
    atlasDebug("map fiber line_points=" + std::to_string(validatedFiber.linePoints.size()) +
               " control_points=" + std::to_string(validatedFiber.controlPoints.size()) +
               " seed_index=" + std::to_string(seedIndex));
    std::vector<std::vector<ProjectionHit>> hitsByLinePoint(validatedFiber.linePoints.size());
    for (size_t i = 0; i < validatedFiber.linePoints.size(); ++i) {
        const auto sample = normalSampler.sampleNormal(validatedFiber.linePoints[i]);
        if (!sample.valid || !validNormal(sample.normal)) {
            atlasDebug("line_point[" + std::to_string(i) + "] invalid_normal point=" +
                       vecString(validatedFiber.linePoints[i]));
            if (static_cast<int>(i) == seedIndex) {
                throw std::runtime_error("No valid normal at atlas seed point");
            }
            continue;
        }
        hitsByLinePoint[i] = projectPointToSurfacesAdaptive(
            validatedFiber.linePoints[i], sample.normal, baseCandidates, baseIndex, options.rayHalfLength);
        if (hitsByLinePoint[i].empty()) {
            atlasDebug("line_point[" + std::to_string(i) + "] no_hits point=" +
                       vecString(validatedFiber.linePoints[i]) + " normal=" + vecString(sample.normal));
        }
    }

    if (hitsByLinePoint[seedIndex].empty()) {
        throw std::runtime_error("failed to project atlas seed point onto the base shell");
    }

    std::vector<std::optional<AtlasAnchor>> anchors(validatedFiber.linePoints.size());
    anchors[seedIndex] = anchorFromHit(seedIndex, hitsByLinePoint[seedIndex].front());
    anchors[seedIndex]->atlasU = normalizeAtlasU(anchors[seedIndex]->atlasU, periodColumns);
    atlasDebug("line_point[" + std::to_string(seedIndex) + "] chosen_anchor u=" +
               std::to_string(anchors[seedIndex]->atlasU) + " v=" +
               std::to_string(anchors[seedIndex]->atlasV));

    int mappedFirst = seedIndex;
    int mappedLast = seedIndex;
    for (int i = seedIndex + 1; i < static_cast<int>(validatedFiber.linePoints.size()); ++i) {
        ContinuationRejectDebug rejectDebug;
        const auto chosen = chooseContinuationHit(i,
                                                  hitsByLinePoint[i],
                                                  *anchors[i - 1],
                                                  validatedFiber.linePoints[i - 1],
                                                  validatedFiber.linePoints[i],
                                                  periodColumns,
                                                  atlasNominalStep,
                                                  options.mismatchRatio,
                                                  atlasDebugEnabled() ? &rejectDebug : nullptr);
        if (!chosen) {
            atlasDebug(continuationRejectDebugString(i, rejectDebug));
            break;
        }
        anchors[i] = *chosen;
        mappedLast = i;
        atlasDebug("line_point[" + std::to_string(i) + "] chosen_anchor u=" +
                   std::to_string(anchors[i]->atlasU) + " v=" +
                   std::to_string(anchors[i]->atlasV));
    }
    for (int i = seedIndex - 1; i >= 0; --i) {
        ContinuationRejectDebug rejectDebug;
        const auto chosen = chooseContinuationHit(i,
                                                  hitsByLinePoint[i],
                                                  *anchors[i + 1],
                                                  validatedFiber.linePoints[i + 1],
                                                  validatedFiber.linePoints[i],
                                                  periodColumns,
                                                  atlasNominalStep,
                                                  options.mismatchRatio,
                                                  atlasDebugEnabled() ? &rejectDebug : nullptr);
        if (!chosen) {
            atlasDebug(continuationRejectDebugString(i, rejectDebug));
            break;
        }
        anchors[i] = *chosen;
        mappedFirst = i;
        atlasDebug("line_point[" + std::to_string(i) + "] chosen_anchor u=" +
                   std::to_string(anchors[i]->atlasU) + " v=" +
                   std::to_string(anchors[i]->atlasV));
    }

    FiberMapping mapping;
    mapping.fiberPath = validatedFiber.fiberPath;
    for (int i = mappedFirst; i <= mappedLast; ++i) {
        if (!anchors[static_cast<size_t>(i)]) {
            break;
        }
        mapping.lineAnchors.push_back(*anchors[static_cast<size_t>(i)]);
    }
    atlasDebug("final_line_anchor_count=" + std::to_string(mapping.lineAnchors.size()));
    if (mapping.lineAnchors.size() < 2) {
        throw std::runtime_error("incomplete atlas mapping: produced fewer than two line anchors");
    }

    for (const int lineIndex : validatedFiber.controlLineIndices) {
        if (lineIndex < mappedFirst || lineIndex > mappedLast) {
            continue;
        }
        const auto& anchor = anchors[static_cast<size_t>(lineIndex)];
        if (anchor) {
            mapping.controlAnchors.push_back(*anchor);
        }
    }
    return mapping;
}

Atlas createSingleFiberAtlas(const fs::path& volpkgRoot,
                             const std::string& atlasName,
                             const FiberInput& fiber,
                             const SurfaceCandidate& baseSurface,
                             int zeroWindingColumn,
                             FiberMapping mapping)
{
    if (!baseSurface.surface) {
        throw std::runtime_error("base surface is null");
    }
    Atlas atlas;
    atlas.metadata.name = sanitizeAtlasName(atlasName);
    const std::string baseDirName = sanitizeAtlasName(baseSurface.name) + ".tifxyz";
    atlas.metadata.baseMeshPath = fs::path("base_mesh") / baseDirName;
    atlas.metadata.sourceBaseMeshPath = fs::relative(baseSurface.path, volpkgRoot);
    atlas.metadata.zeroWindingColumn = zeroWindingColumn;
    atlas.metadata.seedLineIndex = seedLineIndexForFiber(fiber);
    auto seedIt = std::find_if(mapping.lineAnchors.begin(),
                               mapping.lineAnchors.end(),
                               [&atlas](const AtlasAnchor& anchor) {
                                   return anchor.sourceIndex == atlas.metadata.seedLineIndex;
                               });
    if (seedIt != mapping.lineAnchors.end()) {
        atlas.metadata.seedAtlasU = seedIt->atlasU;
        atlas.metadata.seedAtlasV = seedIt->atlasV;
    }
    atlas.fibers.push_back(std::move(mapping));
    return atlas;
}

} // namespace vc::atlas
