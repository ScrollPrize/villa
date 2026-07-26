#include "vc/atlas/Atlas.hpp"

#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/lasagna/Manifest.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "vc/lasagna/LineModel.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <numeric>
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
constexpr int kAtlasMetadataVersion = 5;
constexpr std::array<const char*, 5> kCoordinateMetadataKeys{
    "vc_open_data_coordinate_space",
    "vc_open_data_source_path",
    "vc_open_data_source_coordinate_level",
    "vc_open_data_source_coordinate_scale_factor",
    "vc_open_data_source_original_resolution",
};

void loadCoordinateMetadata(const nlohmann::json& source, AtlasMetadata& target)
{
    target.coordinateMetadata = nlohmann::json::object();
    for (const char* key : kCoordinateMetadataKeys) {
        if (source.contains(key))
            target.coordinateMetadata[key] = source.at(key);
    }
}

void saveCoordinateMetadata(const AtlasMetadata& source, nlohmann::json& target)
{
    if (!source.coordinateMetadata.is_object())
        return;
    for (const char* key : kCoordinateMetadataKeys) {
        if (source.coordinateMetadata.contains(key))
            target[key] = source.coordinateMetadata.at(key);
    }
}
constexpr int kFiberMappingVersion = 4;

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

void validateMappingControlAnchorsAgainstFiber(const FiberMapping& mapping,
                                               const FiberInput& fiber,
                                               const fs::path& mappingPath)
{
    std::unordered_map<int, size_t> controlRowByLineIndex;
    controlRowByLineIndex.reserve(fiber.controlLineIndices.size());
    for (size_t controlIndex = 0; controlIndex < fiber.controlLineIndices.size(); ++controlIndex) {
        controlRowByLineIndex.emplace(fiber.controlLineIndices[controlIndex], controlIndex);
    }

    const double maxDistanceSq = kControlPointMatchEpsilon * kControlPointMatchEpsilon;
    for (size_t anchorIndex = 0; anchorIndex < mapping.controlAnchors.size(); ++anchorIndex) {
        const AtlasAnchor& anchor = mapping.controlAnchors[anchorIndex];
        const auto rowIt = controlRowByLineIndex.find(anchor.sourceIndex);
        if (rowIt == controlRowByLineIndex.end()) {
            throw std::runtime_error(
                "atlas fiber mapping " + mappingPath.string() +
                " control_anchors[" + std::to_string(anchorIndex) +
                "] source_index " + std::to_string(anchor.sourceIndex) +
                " does not identify a fiber control point line_points index; rebuild required");
        }
        const cv::Vec3d& expected = fiber.controlPoints[rowIt->second];
        if (!finitePoint(anchor.world) ||
            squaredDistance(anchor.world, expected) > maxDistanceSq) {
            throw std::runtime_error(
                "atlas fiber mapping " + mappingPath.string() +
                " control_anchors[" + std::to_string(anchorIndex) +
                "] world does not match source fiber control point; rebuild required");
        }
    }
}

fs::path inferVolpkgRootFromAtlasDir(const fs::path& atlasDir)
{
    const fs::path normalized = atlasDir.lexically_normal();
    if (normalized.parent_path().filename() == "atlases") {
        return normalized.parent_path().parent_path();
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

fs::path resolveAtlasFiberPath(const fs::path& atlasDir,
                               const fs::path& volpkgRoot,
                               const fs::path& fiberPathRoot,
                               const fs::path& jsonPath)
{
    if (jsonPath.empty() || jsonPath.is_absolute() || fiberPathRoot.empty()) {
        return resolveAtlasRelativePath(atlasDir, volpkgRoot, jsonPath);
    }

    std::vector<fs::path> candidates;
    candidates.push_back((fiberPathRoot / jsonPath).lexically_normal());

    auto relativeIt = jsonPath.begin();
    if (relativeIt != jsonPath.end() && relativeIt->string() == "fibers") {
        fs::path withoutFibers;
        ++relativeIt;
        for (; relativeIt != jsonPath.end(); ++relativeIt) {
            withoutFibers /= *relativeIt;
        }
        if (!withoutFibers.empty()) {
            candidates.push_back((fiberPathRoot / withoutFibers).lexically_normal());
        }
    }

    candidates.push_back((fiberPathRoot / jsonPath.filename()).lexically_normal());
    for (const auto& candidate : candidates) {
        if (fs::is_regular_file(candidate)) {
            return candidate;
        }
    }
    return candidates.front();
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
    loadCoordinateMetadata(metadata, atlas.metadata);

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

AtlasAnchor anchorFromHit(int sourceIndex, const cv::Vec3d& sourcePoint, const ProjectionHit& hit)
{
    AtlasAnchor anchor;
    anchor.sourceIndex = sourceIndex;
    anchor.world = sourcePoint;
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
            AtlasAnchor candidate = anchorFromHit(sourceIndex, linePoint, hit);
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
    saveCoordinateMetadata(metadata, metadataJson);
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
    return Atlas::load(atlasDir, inferVolpkgRootFromAtlasDir(atlasDir));
}

Atlas Atlas::load(const fs::path& atlasDir, const fs::path& volpkgRootIn)
{
    return Atlas::load(atlasDir, volpkgRootIn, {});
}

Atlas Atlas::load(const fs::path& atlasDir,
                  const fs::path& volpkgRootIn,
                  const fs::path& fiberPathRoot)
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
    loadCoordinateMetadata(metadata, atlas.metadata);

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
            if (mapping.fiberPath.empty()) {
                throw std::runtime_error("atlas fiber mapping " + mappingPath.string() +
                                         " references missing fiber path; rebuild required");
            }
            const fs::path volpkgRoot = volpkgRootIn.empty()
                ? inferVolpkgRootFromAtlasDir(atlasDir)
                : volpkgRootIn;
            if (volpkgRoot.empty() && fiberPathRoot.empty()) {
                throw std::runtime_error("cannot validate atlas fiber mapping " +
                                         mappingPath.string() +
                                         " without a volume package root; rebuild required");
            }
            const fs::path fiberPath = resolveAtlasFiberPath(
                atlasDir, volpkgRoot, fiberPathRoot, mapping.fiberPath);
            if (!fs::is_regular_file(fiberPath)) {
                throw std::runtime_error("atlas fiber mapping " + mappingPath.string() +
                                         " references missing fiber path: " +
                                         mapping.fiberPath.generic_string() +
                                         "; rebuild required");
            }
            const FiberInput sourceFiber = loadSourceFiberInput(fiberPath, mapping.fiberPath);
            validateMappingControlAnchorsAgainstFiber(mapping, sourceFiber, mappingPath);
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

AtlasBaseMappingContext atlasBaseMappingContextFromSurface(
    std::shared_ptr<QuadSurface> baseSurface)
{
    if (!baseSurface) {
        throw std::runtime_error("atlas base mapping context has no base surface");
    }
    const auto* points = baseSurface->rawPointsPtr();
    if (!points || points->empty() || points->cols <= 0) {
        throw std::runtime_error("atlas base mesh has no valid grid");
    }
    AtlasBaseMappingContext context;
    context.baseSurface = std::move(baseSurface);
    context.baseIndex = std::make_shared<SurfacePatchIndex>();
    context.baseIndex->rebuild({context.baseSurface});
    return context;
}

AtlasBaseMappingContext loadAtlasBaseMappingContext(const fs::path& atlasDir,
                                                    const Atlas& atlas)
{
    if (atlas.metadata.baseMeshPath.empty()) {
        throw std::runtime_error("cannot load atlas base mapping context without base_mesh_path");
    }
    return atlasBaseMappingContextFromSurface(
        std::make_shared<QuadSurface>(atlasDir / atlas.metadata.baseMeshPath));
}

Atlas rebuildAtlasFromSourceFibers(const fs::path& atlasDir,
                                   const fs::path& volpkgRoot,
                                   const vc::lasagna::NormalSampler& normalSampler,
                                   const LineMappingOptions& options)
{
    const Atlas legacy = loadAtlasContextForRebuild(atlasDir);
    auto context = loadAtlasBaseMappingContext(atlasDir, legacy);
    return rebuildAtlasFromSourceFibers(
        atlasDir, volpkgRoot, *context.baseSurface, *context.baseIndex, normalSampler, options);
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
    std::vector<FiberInput> rebuiltInputs;
    rebuiltInputs.reserve(legacy.fibers.size());

    for (const auto& oldMapping : legacy.fibers) {
        if (oldMapping.fiberPath.empty()) {
            throw std::runtime_error("cannot rebuild atlas mapping with missing fiber path");
        }
        const fs::path fiberPath = resolveAtlasRelativePath(
            atlasDir, volpkgRoot, oldMapping.fiberPath);
        const FiberInput input = loadSourceFiberInput(fiberPath, oldMapping.fiberPath);
        rebuilt.fibers.push_back(
            mapFiberToBaseSurface(input, baseSurface, baseIndex, normalSampler, options));
        rebuiltInputs.push_back(input);
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
    if (const auto* predSampler =
            dynamic_cast<const vc::lasagna::LasagnaNormalSampler*>(&normalSampler);
        predSampler && predSampler->hasPredDtChannel()) {
        for (size_t i = 0; i < rebuilt.fibers.size() && i < rebuiltInputs.size(); ++i) {
            (void)ensureAtlasPredSnapSet(atlasDir,
                                         rebuiltInputs[i],
                                         rebuilt.fibers[i],
                                         baseSurface,
                                         *predSampler);
        }
    }
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

std::string atlasPredSnapControlPointKey(const cv::Vec3d& point)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17)
        << point[0] << ',' << point[1] << ',' << point[2];
    return out.str();
}

fs::path atlasPredSnapAttachmentPath(const fs::path& atlasDir, const fs::path& fiberPath)
{
    std::string safe = fiberPath.stem().empty()
        ? std::string("fiber")
        : fiberPath.stem().string();
    safe = sanitizeAtlasName(safe);
    return atlasDir / "attachments" / "pred_snap_points" / (safe + ".json");
}

bool atlasPredDtIsInside(double predDtValue, double threshold)
{
    if (!std::isfinite(threshold)) {
        threshold = 110.0;
    }
    threshold = std::clamp(threshold, 0.0, 255.0);
    return std::isfinite(predDtValue) && predDtValue >= threshold;
}

namespace {

std::string predSnapSourceString(AtlasPredSnapSource source)
{
    switch (source) {
    case AtlasPredSnapSource::Manual:
        return "manual";
    case AtlasPredSnapSource::Optimized:
        return "optimized";
    case AtlasPredSnapSource::Auto:
    default:
        return "auto";
    }
}

AtlasPredSnapSource predSnapSourceFromString(const std::string& value)
{
    if (value == "manual") {
        return AtlasPredSnapSource::Manual;
    }
    if (value == "optimized") {
        return AtlasPredSnapSource::Optimized;
    }
    return AtlasPredSnapSource::Auto;
}

std::string predSnapDirectionString(AtlasPredSnapDirection direction)
{
    return direction == AtlasPredSnapDirection::Inside ? "inside" : "outside";
}

AtlasPredSnapDirection predSnapDirectionFromString(const std::string& value)
{
    return value == "inside" ? AtlasPredSnapDirection::Inside : AtlasPredSnapDirection::Outside;
}

void setPredSnapStatus(AtlasPredSnapPoint& point,
                       std::string status,
                       std::string reason)
{
    point.status = std::move(status);
    point.statusReason = std::move(reason);
}

nlohmann::json predSnapCandidateJson(const AtlasPredSnapCandidate& candidate)
{
    nlohmann::json root;
    root["point"] = pointJson(candidate.point);
    if (candidate.predDtValue) {
        root["pred_dt_value"] = *candidate.predDtValue;
    }
    if (candidate.direction) {
        root["direction"] = predSnapDirectionString(*candidate.direction);
    }
    if (candidate.windingDistance) {
        root["winding_distance"] = *candidate.windingDistance;
    }
    return root;
}

AtlasPredSnapCandidate predSnapCandidateFromJson(const nlohmann::json& root)
{
    AtlasPredSnapCandidate candidate;
    candidate.point = pointFromJson(root.at("point"));
    if (root.contains("pred_dt_value") && root["pred_dt_value"].is_number()) {
        candidate.predDtValue = root["pred_dt_value"].get<double>();
    }
    if (root.contains("direction") && root["direction"].is_string()) {
        candidate.direction = predSnapDirectionFromString(root["direction"].get<std::string>());
    }
    if (root.contains("winding_distance") && root["winding_distance"].is_number()) {
        candidate.windingDistance = root["winding_distance"].get<double>();
    }
    return candidate;
}

nlohmann::json predSnapPointJson(const AtlasPredSnapPoint& point)
{
    nlohmann::json root;
    root["fiber_path"] = point.fiberPath.generic_string();
    if (point.sourceIndex) {
        root["source_index"] = *point.sourceIndex;
    }
    root["control_point"] = pointJson(point.controlPoint);
    if (point.predSnapPoint) {
        root["pred_snap_point"] = pointJson(*point.predSnapPoint);
    } else {
        root["pred_snap_point"] = nullptr;
    }
    if (!point.candidates.empty()) {
        root["candidates"] = nlohmann::json::array();
        for (const auto& candidate : point.candidates) {
            root["candidates"].push_back(predSnapCandidateJson(candidate));
        }
    }
    if (point.selectedCandidateIndex) {
        root["selected_candidate_index"] = *point.selectedCandidateIndex;
    }
    root["source"] = predSnapSourceString(point.source);
    if (!point.status.empty()) {
        root["status"] = point.status;
    }
    if (!point.statusReason.empty()) {
        root["status_reason"] = point.statusReason;
    }
    if (point.predDtValue) {
        root["pred_dt_value"] = *point.predDtValue;
    }
    if (point.direction) {
        root["direction"] = predSnapDirectionString(*point.direction);
    }
    if (point.weightedFirstHitWindingDistance) {
        root["weighted_first_hit_winding_distance"] =
            *point.weightedFirstHitWindingDistance;
    }
    if (point.searchNormal) {
        root["search_normal"] = pointJson(*point.searchNormal);
    }
    if (!point.generatedAtUtc.empty()) {
        root["generated_at_utc"] = point.generatedAtUtc;
    }
    return root;
}

AtlasPredSnapPoint predSnapPointFromJson(const nlohmann::json& root,
                                         const fs::path& fallbackFiberPath)
{
    AtlasPredSnapPoint point;
    point.fiberPath = root.value("fiber_path", fallbackFiberPath.generic_string());
    if (root.contains("source_index") && root["source_index"].is_number_integer()) {
        point.sourceIndex = root["source_index"].get<int>();
    }
    point.controlPoint = pointFromJson(root.at("control_point"));
    if (root.contains("pred_snap_point") && !root["pred_snap_point"].is_null()) {
        point.predSnapPoint = pointFromJson(root["pred_snap_point"]);
    }
    if (root.contains("candidates") && root["candidates"].is_array()) {
        for (const auto& candidate : root["candidates"]) {
            if (candidate.is_object() && candidate.contains("point")) {
                point.candidates.push_back(predSnapCandidateFromJson(candidate));
            }
        }
    }
    if (root.contains("selected_candidate_index") &&
        root["selected_candidate_index"].is_number_integer()) {
        point.selectedCandidateIndex = root["selected_candidate_index"].get<int>();
    }
    point.source = predSnapSourceFromString(root.value("source", std::string("auto")));
    point.status = root.value("status", std::string{});
    point.statusReason = root.value("status_reason", std::string{});
    if (root.contains("pred_dt_value") && root["pred_dt_value"].is_number()) {
        point.predDtValue = root["pred_dt_value"].get<double>();
    }
    if (root.contains("direction") && root["direction"].is_string()) {
        point.direction = predSnapDirectionFromString(root["direction"].get<std::string>());
    }
    if (root.contains("weighted_first_hit_winding_distance") &&
        root["weighted_first_hit_winding_distance"].is_number()) {
        point.weightedFirstHitWindingDistance =
            root["weighted_first_hit_winding_distance"].get<double>();
    }
    if (root.contains("search_normal") && root["search_normal"].is_array()) {
        point.searchNormal = pointFromJson(root["search_normal"]);
    }
    point.generatedAtUtc = root.value("generated_at_utc", std::string{});
    if (point.candidates.empty() && point.predSnapPoint) {
        AtlasPredSnapCandidate candidate;
        candidate.point = *point.predSnapPoint;
        candidate.predDtValue = point.predDtValue;
        candidate.direction = point.direction;
        candidate.windingDistance = point.weightedFirstHitWindingDistance;
        point.candidates.push_back(candidate);
        point.selectedCandidateIndex = 0;
    }
    return point;
}

cv::Vec3d normalizedPredSnapNormal(const cv::Vec3d& normal)
{
    if (!finitePoint(normal)) {
        return {0.0, 0.0, 0.0};
    }
    const double n = norm(normal);
    if (!(n > kEpsilon) || !std::isfinite(n)) {
        return {0.0, 0.0, 0.0};
    }
    return normal * (1.0 / n);
}

struct PredSnapTraceSample {
    cv::Vec3d point{0.0, 0.0, 0.0};
    double windingDistance = 0.0;
    std::optional<double> predDt;
};

std::vector<PredSnapTraceSample> tracePredSnapDirection(
    const cv::Vec3d& start,
    const cv::Vec3d& unitDirection,
    double maxWinding,
    const AtlasPredSnapSampling& sampling)
{
    std::vector<PredSnapTraceSample> samples;
    const double stepVx = std::isfinite(sampling.predDtStepVx) && sampling.predDtStepVx > 0.0
        ? sampling.predDtStepVx
        : 0.05;
    if (!sampling.samplePredDt || !sampling.windingDistance ||
        !finitePoint(start) || !validNormal(unitDirection) ||
        !(maxWinding > 0.0) || !std::isfinite(maxWinding)) {
        return samples;
    }

    constexpr int kMaxSteps = 1024;
    cv::Vec3d previous = start;
    double accumulated = 0.0;
    for (int step = 1; step <= kMaxSteps && accumulated <= maxWinding + 1.0e-9; ++step) {
        const cv::Vec3d current = start + unitDirection * (stepVx * static_cast<double>(step));
        const double segmentWinding = sampling.windingDistance(previous, current, stepVx);
        if (!std::isfinite(segmentWinding) || segmentWinding < 0.0) {
            break;
        }
        accumulated += segmentWinding;
        if (accumulated > maxWinding + 1.0e-9) {
            break;
        }
        samples.push_back({current, accumulated, sampling.samplePredDt(current)});
        previous = current;
        if (segmentWinding <= kEpsilon && step > 16) {
            break;
        }
    }
    return samples;
}

std::optional<PredSnapTraceSample> firstInsidePredSnapHit(
    const std::vector<PredSnapTraceSample>& samples,
    double predDtThreshold)
{
    for (const auto& sample : samples) {
        if (sample.predDt && atlasPredDtIsInside(*sample.predDt, predDtThreshold)) {
            return sample;
        }
    }
    return std::nullopt;
}

std::string roundedPredSnapCandidateKey(const cv::Vec3d& point)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::llround(point[0] * 1000000.0) << ','
        << std::llround(point[1] * 1000000.0) << ','
        << std::llround(point[2] * 1000000.0);
    return out.str();
}

PredSnapTraceSample refinePredSnapCandidateAlongNormal(
    const PredSnapTraceSample& seed,
    const cv::Vec3d& unitNormal,
    const AtlasPredSnapSampling& sampling)
{
    const double stepVx = std::isfinite(sampling.predDtStepVx) && sampling.predDtStepVx > 0.0
        ? sampling.predDtStepVx
        : 0.05;
    const double predDtThreshold = std::isfinite(sampling.predDtThreshold)
        ? sampling.predDtThreshold
        : 110.0;
    if (!sampling.samplePredDt || !validNormal(unitNormal) || !finitePoint(seed.point) ||
        !seed.predDt || !atlasPredDtIsInside(*seed.predDt, predDtThreshold)) {
        return seed;
    }

    constexpr int kMaxSteps = 1024;
    PredSnapTraceSample current = seed;
    for (int step = 0; step < kMaxSteps; ++step) {
        PredSnapTraceSample best = current;
        for (const double sign : {1.0, -1.0}) {
            const cv::Vec3d point = current.point + unitNormal * (sign * stepVx);
            const auto predDt = sampling.samplePredDt(point);
            if (!predDt || !atlasPredDtIsInside(*predDt, predDtThreshold) ||
                (best.predDt && *predDt <= *best.predDt)) {
                continue;
            }
            best.point = point;
            best.predDt = predDt;
            best.windingDistance = seed.windingDistance;
        }
        if (norm(best.point - current.point) <= kEpsilon) {
            break;
        }
        current = best;
    }
    return current;
}

std::optional<cv::Vec3d> baseNormalForAnchor(const AtlasAnchor& anchor,
                                             const FiberMapping& mapping,
                                             const QuadSurface& baseSurface,
                                             double outwardSign = 1.0)
{
    const auto* points = baseSurface.rawPointsPtr();
    if (!points || points->empty()) {
        return std::nullopt;
    }
    const int periodColumns = atlasHorizontalPeriodColumns(baseSurface);
    const double u = normalizeAtlasU(actualAtlasU(anchor, mapping, periodColumns),
                                     periodColumns);
    const double v = anchor.atlasV;
    if (!std::isfinite(u) || !std::isfinite(v) ||
        u < 0.0 || v < 0.0 ||
        u >= static_cast<double>(points->cols) ||
        v >= static_cast<double>(points->rows)) {
        return std::nullopt;
    }
    const cv::Vec3f normal = grid_normal(*points, cv::Vec3f(static_cast<float>(u),
                                                            static_cast<float>(v),
                                                            0.0f));
    if (!finitePoint(normal) || cv::norm(normal) <= 1.0e-6f) {
        return std::nullopt;
    }
    return normalizedPredSnapNormal(toVec3d(normal) * outwardSign);
}

double ringPerimeterWithNormalOffset(const cv::Mat_<cv::Vec3f>& points,
                                     int row,
                                     int periodColumns,
                                     double offset)
{
    double perimeter = 0.0;
    bool havePrevious = false;
    cv::Vec3d first{0.0, 0.0, 0.0};
    cv::Vec3d previous{0.0, 0.0, 0.0};
    for (int col = 0; col < periodColumns; ++col) {
        const cv::Vec3f p = points(row, col);
        if (!finitePoint(p)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const cv::Vec3f n = grid_normal(points, cv::Vec3f(static_cast<float>(col),
                                                          static_cast<float>(row),
                                                          0.0f));
        if (!finitePoint(n) || cv::norm(n) <= 1.0e-6f) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const cv::Vec3d current = toVec3d(p) + normalizedPredSnapNormal(toVec3d(n)) * offset;
        if (!havePrevious) {
            first = current;
            havePrevious = true;
        } else {
            perimeter += norm(current - previous);
        }
        previous = current;
    }
    if (!havePrevious) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    perimeter += norm(first - previous);
    return perimeter;
}

double atlasBaseNormalOutwardSign(const QuadSurface& baseSurface)
{
    const auto* points = baseSurface.rawPointsPtr();
    if (!points || points->rows < 5 || points->cols < 5) {
        return 1.0;
    }
    const int periodColumns = atlasHorizontalPeriodColumns(baseSurface);
    if (periodColumns < 4) {
        return 1.0;
    }
    const int row = points->rows / 2;
    double edgeSum = 0.0;
    int edgeCount = 0;
    for (int col = 0; col < periodColumns; ++col) {
        const cv::Vec3f a = (*points)(row, col);
        const cv::Vec3f b = (*points)(row, (col + 1) % periodColumns);
        if (!finitePoint(a) || !finitePoint(b)) {
            continue;
        }
        const double edge = norm(toVec3d(b) - toVec3d(a));
        if (std::isfinite(edge) && edge > kEpsilon) {
            edgeSum += edge;
            ++edgeCount;
        }
    }
    if (edgeCount == 0) {
        return 1.0;
    }
    const double eps = std::clamp(edgeSum / static_cast<double>(edgeCount), 0.5, 4.0);
    const double plus = ringPerimeterWithNormalOffset(*points, row, periodColumns, eps);
    const double minus = ringPerimeterWithNormalOffset(*points, row, periodColumns, -eps);
    if (!std::isfinite(plus) || !std::isfinite(minus) ||
        std::abs(plus - minus) <= 1.0e-6) {
        return 1.0;
    }
    return plus > minus ? 1.0 : -1.0;
}

} // namespace

std::vector<AtlasPredSnapCandidate> findAtlasPredSnapCandidates(
    const cv::Vec3d& controlPoint,
    const cv::Vec3d& alignedNormal,
    const AtlasPredSnapSampling& sampling)
{
    std::vector<AtlasPredSnapCandidate> candidates;
    if (!sampling.samplePredDt || !sampling.windingDistance ||
        !finitePoint(controlPoint) || !validNormal(alignedNormal)) {
        return candidates;
    }
    const cv::Vec3d normal = normalizedPredSnapNormal(alignedNormal);
    if (!validNormal(normal)) {
        return candidates;
    }
    const double predDtThreshold = std::isfinite(sampling.predDtThreshold)
        ? sampling.predDtThreshold
        : 110.0;

    std::unordered_set<std::string> seen;
    auto addCandidate = [&](const PredSnapTraceSample& hit, double windingDistance) {
        if (!finitePoint(hit.point)) {
            return;
        }
        const std::string key = roundedPredSnapCandidateKey(hit.point);
        if (!seen.insert(key).second) {
            return;
        }
        const PredSnapTraceSample refined =
            refinePredSnapCandidateAlongNormal(hit, normal, sampling);
        AtlasPredSnapCandidate candidate;
        candidate.point = refined.point;
        candidate.predDtValue = refined.predDt;
        candidate.direction = (refined.point - controlPoint).dot(normal) < 0.0
            ? AtlasPredSnapDirection::Inside
            : AtlasPredSnapDirection::Outside;
        candidate.windingDistance = windingDistance;
        candidates.push_back(std::move(candidate));
    };

    const auto startPredDt = sampling.samplePredDt(controlPoint);
    if (startPredDt && atlasPredDtIsInside(*startPredDt, predDtThreshold)) {
        addCandidate({controlPoint, 0.0, startPredDt}, 0.0);
        return candidates;
    }

    constexpr double kCandidateWindingLimit = 1.0;
    if (const auto outwardHit =
            firstInsidePredSnapHit(tracePredSnapDirection(
                controlPoint, normal, kCandidateWindingLimit, sampling), predDtThreshold)) {
        addCandidate(*outwardHit, outwardHit->windingDistance);
    }
    if (const auto inwardHit =
            firstInsidePredSnapHit(tracePredSnapDirection(
                controlPoint, -normal, kCandidateWindingLimit, sampling), predDtThreshold)) {
        addCandidate(*inwardHit, inwardHit->windingDistance);
    }
    return candidates;
}

AtlasPredSnapSet generateAtlasPredSnapSet(const FiberInput& fiber,
                                          const FiberMapping& mapping,
                                          const QuadSurface& baseSurface,
                                          const AtlasPredSnapSampling& sampling)
{
    AtlasPredSnapSet set;
    set.fiberPath = fiber.fiberPath.empty() ? mapping.fiberPath : fiber.fiberPath;
    set.points.reserve(fiber.controlPoints.size());
    const double baseOutwardSign = atlasBaseNormalOutwardSign(baseSurface);
    for (size_t controlIndex = 0; controlIndex < fiber.controlPoints.size(); ++controlIndex) {
        const cv::Vec3d& controlPoint = fiber.controlPoints[controlIndex];
        AtlasPredSnapPoint point;
        point.fiberPath = set.fiberPath;
        point.controlPoint = controlPoint;
        point.sourceIndex = controlIndex < fiber.controlLineIndices.size()
            ? std::optional<int>(fiber.controlLineIndices[controlIndex])
            : std::optional<int>(static_cast<int>(controlIndex));
        point.source = AtlasPredSnapSource::Auto;

        const int controlSourceIndex =
            controlIndex < fiber.controlLineIndices.size()
                ? fiber.controlLineIndices[controlIndex]
                : static_cast<int>(controlIndex);
        const auto anchorIt = std::find_if(
            mapping.controlAnchors.begin(),
            mapping.controlAnchors.end(),
            [controlSourceIndex](const AtlasAnchor& anchor) {
                return anchor.sourceIndex == controlSourceIndex;
            });
        if (anchorIt == mapping.controlAnchors.end()) {
            setPredSnapStatus(point,
                              "missing_control_anchor",
                              "mapping has no control anchor for source_index " +
                                  std::to_string(controlSourceIndex));
            set.points.push_back(std::move(point));
            continue;
        }
        if (!sampling.sampleNormal) {
            setPredSnapStatus(point,
                              "normal_sampling_unavailable",
                              "no normal sampler is available for pred-snap candidate generation");
            set.points.push_back(std::move(point));
            continue;
        }

        const auto normalSample = sampling.sampleNormal(controlPoint);
        const auto baseNormal = baseNormalForAnchor(*anchorIt,
                                                    mapping,
                                                    baseSurface,
                                                    baseOutwardSign);
        if (!normalSample.valid || !validNormal(normalSample.normal)) {
            std::string reason = "invalid Lasagna normal at control point";
            if (!normalSample.reason.empty()) {
                reason += ": " + normalSample.reason;
            }
            setPredSnapStatus(point, "invalid_lasagna_normal", std::move(reason));
            set.points.push_back(std::move(point));
            continue;
        }
        if (!baseNormal) {
            setPredSnapStatus(point,
                              "invalid_atlas_base_normal",
                              "could not sample a valid atlas base normal at the control anchor");
            set.points.push_back(std::move(point));
            continue;
        }

        cv::Vec3d alignedNormal = normalizedPredSnapNormal(normalSample.normal);
        if (alignedNormal.dot(*baseNormal) < 0.0) {
            alignedNormal *= -1.0;
        }
        point.searchNormal = alignedNormal;
        const auto startPredDt = sampling.samplePredDt
            ? sampling.samplePredDt(controlPoint)
            : std::optional<double>{};
        const double predDtThreshold = std::isfinite(sampling.predDtThreshold)
            ? sampling.predDtThreshold
            : 110.0;
        const bool startsInside =
            startPredDt && atlasPredDtIsInside(*startPredDt, predDtThreshold);
        point.candidates = findAtlasPredSnapCandidates(controlPoint, alignedNormal, sampling);
        const size_t requiredCandidates = 1U;
        if (point.candidates.size() < requiredCandidates) {
            std::ostringstream reason;
            reason << "candidate search produced " << point.candidates.size()
                   << " candidate(s), but at least one usable candidate is required; "
                   << "control point starts "
                   << (startsInside ? "inside" : "outside")
                   << " the acceptable pred-dt range";
            if (!startPredDt) {
                reason << "; start pred_dt unavailable";
            } else {
                reason << "; start pred_dt=" << *startPredDt
                       << " threshold=" << predDtThreshold;
            }
            setPredSnapStatus(point, "insufficient_candidates_none", reason.str());
        } else {
            std::ostringstream reason;
            reason << "candidate search produced " << point.candidates.size()
                   << " usable candidate(s); at least one is required; control point starts "
                   << (startsInside ? "inside" : "outside")
                   << " the acceptable pred-dt range";
            setPredSnapStatus(point,
                              point.candidates.size() >= 2 ? "ready_two_sided" : "ready_single",
                              reason.str());
        }
        set.points.push_back(std::move(point));
    }
    return set;
}

AtlasPredSnapSet generateAtlasPredSnapSet(const FiberInput& fiber,
                                          const FiberMapping& mapping,
                                          const QuadSurface& baseSurface,
                                          const vc::lasagna::LasagnaNormalSampler& sampler)
{
    AtlasPredSnapSampling sampling;
    sampling.sampleNormal = [&sampler](const cv::Vec3d& point) {
        return sampler.sampleNormal(point);
    };
    sampling.samplePredDt = [&sampler](const cv::Vec3d& point) {
        return sampler.samplePredDt(point);
    };
    sampling.windingDistance = [&sampler](const cv::Vec3d& a,
                                          const cv::Vec3d& b,
                                          double stepVx) {
        return sampler.windingDistance(a, b, stepVx);
    };
    const auto predDtSpacing = sampler.predDtSpacing();
    if (!predDtSpacing || !std::isfinite(*predDtSpacing) || *predDtSpacing <= 0.0) {
        throw std::runtime_error("Lasagna pred_dt channel has no valid spacing");
    }
    sampling.predDtStepVx = 0.5 * *predDtSpacing;
    return generateAtlasPredSnapSet(fiber, mapping, baseSurface, sampling);
}

AtlasPredSnapSet loadAtlasPredSnapSet(const fs::path& attachmentPath)
{
    AtlasPredSnapSet set;
    if (!fs::exists(attachmentPath)) {
        return set;
    }
    const auto root = readJsonFile(attachmentPath);
    if (root.value("type", std::string{}) != "vc3d_atlas_pred_snap_points" ||
        root.value("version", 0) != 1) {
        throw std::runtime_error("unsupported atlas pred-snap attachment: " +
                                 attachmentPath.string());
    }
    set.fiberPath = root.value("fiber_path", std::string{});
    if (root.contains("entries") && root["entries"].is_object()) {
        for (const auto& item : root["entries"].items()) {
            (void)item;
            try {
                set.points.push_back(predSnapPointFromJson(item.value(), set.fiberPath));
            } catch (const std::exception&) {
                throw;
            }
        }
    }
    return set;
}

void saveAtlasPredSnapSet(const fs::path& attachmentPath, const AtlasPredSnapSet& set)
{
    nlohmann::json root;
    root["type"] = "vc3d_atlas_pred_snap_points";
    root["version"] = 1;
    root["fiber_path"] = set.fiberPath.generic_string();
    root["entries"] = nlohmann::json::object();
    for (const auto& point : set.points) {
        root["entries"][atlasPredSnapControlPointKey(point.controlPoint)] =
            predSnapPointJson(point);
    }
    writeJsonFile(attachmentPath, root);
}

AtlasPredSnapSet mergeAtlasPredSnapSetByControlPoint(AtlasPredSnapSet existing,
                                                     const AtlasPredSnapSet& generated)
{
    std::unordered_map<std::string, AtlasPredSnapPoint> byKey;
    byKey.reserve(existing.points.size());
    for (auto& point : existing.points) {
        byKey[atlasPredSnapControlPointKey(point.controlPoint)] = std::move(point);
    }

    AtlasPredSnapSet merged;
    merged.fiberPath = !generated.fiberPath.empty() ? generated.fiberPath : existing.fiberPath;
    merged.points.reserve(generated.points.size());
    for (const auto& generatedPoint : generated.points) {
        const std::string key = atlasPredSnapControlPointKey(generatedPoint.controlPoint);
        auto it = byKey.find(key);
        if (it != byKey.end() && it->second.source == AtlasPredSnapSource::Manual) {
            AtlasPredSnapPoint point = std::move(it->second);
            point.fiberPath = merged.fiberPath;
            merged.points.push_back(std::move(point));
        } else {
            AtlasPredSnapPoint point = generatedPoint;
            if (it != byKey.end() &&
                it->second.source == AtlasPredSnapSource::Optimized &&
                it->second.predSnapPoint) {
                for (size_t i = 0; i < point.candidates.size(); ++i) {
                    if (norm(point.candidates[i].point - *it->second.predSnapPoint) <=
                        kControlPointMatchEpsilon) {
                        point.selectedCandidateIndex = static_cast<int>(i);
                        point.predSnapPoint = point.candidates[i].point;
                        point.source = AtlasPredSnapSource::Optimized;
                        point.predDtValue = point.candidates[i].predDtValue;
                        point.direction = point.candidates[i].direction;
                        point.weightedFirstHitWindingDistance =
                            point.candidates[i].windingDistance;
                        setPredSnapStatus(point,
                                          "optimized",
                                          "previous optimized pred-snap point still matches a regenerated candidate");
                        break;
                    }
                }
            }
            merged.points.push_back(std::move(point));
        }
    }
    return merged;
}

AtlasPredSnapSet ensureAtlasPredSnapSet(const fs::path& atlasDir,
                                        const FiberInput& fiber,
                                        const FiberMapping& mapping,
                                        const QuadSurface& baseSurface,
                                        const AtlasPredSnapSampling& sampling)
{
    const fs::path fiberPath = fiber.fiberPath.empty() ? mapping.fiberPath : fiber.fiberPath;
    const fs::path attachmentPath = atlasPredSnapAttachmentPath(atlasDir, fiberPath);
    AtlasPredSnapSet existing = loadAtlasPredSnapSet(attachmentPath);
    AtlasPredSnapSet generated = generateAtlasPredSnapSet(fiber, mapping, baseSurface, sampling);
    AtlasPredSnapSet merged = mergeAtlasPredSnapSetByControlPoint(std::move(existing), generated);
    saveAtlasPredSnapSet(attachmentPath, merged);
    return merged;
}

AtlasPredSnapSet ensureAtlasPredSnapSet(const fs::path& atlasDir,
                                        const FiberInput& fiber,
                                        const FiberMapping& mapping,
                                        const QuadSurface& baseSurface,
                                        const vc::lasagna::LasagnaNormalSampler& sampler)
{
    AtlasPredSnapSampling sampling;
    sampling.sampleNormal = [&sampler](const cv::Vec3d& point) {
        return sampler.sampleNormal(point);
    };
    sampling.samplePredDt = [&sampler](const cv::Vec3d& point) {
        return sampler.samplePredDt(point);
    };
    sampling.windingDistance = [&sampler](const cv::Vec3d& a,
                                          const cv::Vec3d& b,
                                          double stepVx) {
        return sampler.windingDistance(a, b, stepVx);
    };
    const auto predDtSpacing = sampler.predDtSpacing();
    if (!predDtSpacing || !std::isfinite(*predDtSpacing) || *predDtSpacing <= 0.0) {
        throw std::runtime_error("Lasagna pred_dt channel has no valid spacing");
    }
    sampling.predDtStepVx = 0.5 * *predDtSpacing;
    return ensureAtlasPredSnapSet(atlasDir, fiber, mapping, baseSurface, sampling);
}

AtlasPredSnapSet setManualAtlasPredSnapPoint(const fs::path& atlasDir,
                                             const fs::path& fiberPath,
                                             const cv::Vec3d& controlPoint,
                                             const cv::Vec3d& predSnapPoint,
                                             std::optional<double> predDtValue)
{
    const fs::path attachmentPath = atlasPredSnapAttachmentPath(atlasDir, fiberPath);
    AtlasPredSnapSet set = loadAtlasPredSnapSet(attachmentPath);
    if (set.fiberPath.empty()) {
        set.fiberPath = fiberPath;
    }
    const std::string key = atlasPredSnapControlPointKey(controlPoint);
    auto it = std::find_if(set.points.begin(), set.points.end(), [&key](const AtlasPredSnapPoint& point) {
        return atlasPredSnapControlPointKey(point.controlPoint) == key;
    });
    AtlasPredSnapPoint point;
    point.fiberPath = fiberPath;
    point.controlPoint = controlPoint;
    point.predSnapPoint = predSnapPoint;
    point.source = AtlasPredSnapSource::Manual;
    setPredSnapStatus(point, "manual", "manual pred-snap point");
    point.predDtValue = predDtValue;
    AtlasPredSnapCandidate candidate;
    candidate.point = predSnapPoint;
    candidate.predDtValue = predDtValue;
    point.candidates = {candidate};
    point.selectedCandidateIndex = 0;
    if (it == set.points.end()) {
        set.points.push_back(std::move(point));
    } else {
        *it = std::move(point);
    }
    saveAtlasPredSnapSet(attachmentPath, set);
    return set;
}

AtlasPredSnapAttachmentReport ensureAtlasPredSnapAttachments(
    const fs::path& atlasDir,
    const fs::path& volpkgRootIn,
    const vc::lasagna::LasagnaNormalSampler& sampler)
{
    if (!sampler.hasPredDtChannel()) {
        throw std::runtime_error(
            "selected Lasagna dataset has no pred_dt channel; atlas pred-snap attachments are required");
    }

    const fs::path volpkgRoot = volpkgRootIn.empty()
        ? inferVolpkgRootFromAtlasDir(atlasDir)
        : volpkgRootIn;
    if (volpkgRoot.empty()) {
        throw std::runtime_error("cannot generate atlas pred-snap attachments without a volume package root");
    }

    Atlas atlas = Atlas::load(atlasDir, volpkgRoot);
    if (atlas.metadata.baseMeshPath.empty()) {
        throw std::runtime_error("cannot generate atlas pred-snap attachments without base_mesh_path");
    }
    QuadSurface baseSurface(atlasDir / atlas.metadata.baseMeshPath);
    const auto* points = baseSurface.rawPointsPtr();
    if (!points || points->empty()) {
        throw std::runtime_error("cannot generate atlas pred-snap attachments: base mesh has no valid grid");
    }

    AtlasPredSnapAttachmentReport report;
    for (const auto& mapping : atlas.fibers) {
        if (mapping.fiberPath.empty()) {
            continue;
        }
        ++report.fibersChecked;
        const fs::path attachmentPath =
            atlasPredSnapAttachmentPath(atlasDir, mapping.fiberPath);
        const bool existed = fs::exists(attachmentPath);
        const fs::path fiberPath =
            resolveAtlasRelativePath(atlasDir, volpkgRoot, mapping.fiberPath);
        const FiberInput input = loadSourceFiberInput(fiberPath, mapping.fiberPath);
        (void)ensureAtlasPredSnapSet(atlasDir, input, mapping, baseSurface, sampler);
        if (!existed && fs::exists(attachmentPath)) {
            ++report.attachmentsCreated;
        }
    }
    return report;
}

fs::path atlasPredSnapRankCachePath(const fs::path& atlasDir)
{
    return atlasDir / "attachments" / "pred_snap_rank_cache.json";
}

namespace {

uint64_t fnv1a64(std::string_view text)
{
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char ch : text) {
        h ^= static_cast<uint64_t>(ch);
        h *= 1099511628211ULL;
    }
    return h;
}

std::string hex64(uint64_t value)
{
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

nlohmann::json pointsJson(const std::vector<cv::Vec3d>& points)
{
    nlohmann::json out = nlohmann::json::array();
    for (const auto& point : points) {
        out.push_back(pointJson(point));
    }
    return out;
}

std::string pointsString(const std::vector<cv::Vec3d>& points)
{
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < points.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << vecString(points[i]);
    }
    out << ']';
    return out.str();
}

std::string controlIdFor(const fs::path& fiberPath, int sourceIndex)
{
    return atlasFiberPathKey(fiberPath) + "#" + std::to_string(sourceIndex);
}

std::string termIdFor(size_t a, size_t b)
{
    if (b < a) {
        std::swap(a, b);
    }
    return "term:" + std::to_string(a) + ":" + std::to_string(b);
}

std::optional<size_t> controlIndexForEndpoint(
    const AtlasSnapOptimizationProblem& problem,
    const fs::path& fiberPath,
    int sourceIndex)
{
    const std::string id = controlIdFor(fiberPath, sourceIndex);
    for (size_t i = 0; i < problem.controls.size(); ++i) {
        if (problem.controls[i].id == id) {
            return i;
        }
    }
    return std::nullopt;
}

std::vector<int> bracketingControlSourceIndices(
    const FiberMapping& mapping,
    int sourceIndex)
{
    std::vector<int> indices;
    indices.reserve(mapping.controlAnchors.size());
    for (const auto& anchor : mapping.controlAnchors) {
        indices.push_back(anchor.sourceIndex);
    }
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    if (indices.size() <= 2) {
        return indices;
    }

    auto upper = std::lower_bound(indices.begin(), indices.end(), sourceIndex);
    size_t first = 0;
    size_t second = 1;
    if (upper == indices.begin()) {
        first = 0;
        second = 1;
    } else if (upper == indices.end()) {
        first = indices.size() - 2;
        second = indices.size() - 1;
    } else if (*upper == sourceIndex) {
        second = static_cast<size_t>(upper - indices.begin());
        if (second == 0) {
            first = 0;
            second = 1;
        } else if (second + 1 < indices.size()) {
            first = second;
            ++second;
        } else {
            first = second - 1;
        }
    } else {
        second = static_cast<size_t>(upper - indices.begin());
        first = second - 1;
    }
    return {indices[first], indices[second]};
}

const FiberMapping* mappingForPath(const Atlas& atlas, const fs::path& fiberPath)
{
    const std::string key = atlasFiberPathKey(fiberPath);
    for (const auto& mapping : atlas.fibers) {
        if (atlasFiberPathKey(mapping.fiberPath) == key) {
            return &mapping;
        }
    }
    return nullptr;
}

nlohmann::json controlDebugJson(const AtlasSnapCandidateSet& control)
{
    return {
        {"id", control.id},
        {"fiber_path", control.fiberPath.generic_string()},
        {"source_index", control.sourceIndex},
        {"manual", control.manual},
        {"fixed", control.fixed},
        {"eligible", control.eligible},
        {"status", control.status},
        {"status_reason", control.statusReason},
        {"control_point", pointJson(control.controlPoint)},
        {"candidate_count", control.candidates.size()},
        {"candidates", pointsJson(control.candidates)},
    };
}

nlohmann::json termDebugJson(
    const AtlasSnapPairTerm& term,
    const AtlasSnapOptimizationProblem& problem)
{
    nlohmann::json out = {
        {"term_id", term.id},
        {"first_control_index", term.firstControl},
        {"second_control_index", term.secondControl},
    };
    if (term.firstControl < problem.controls.size()) {
        out["first_control"] = controlDebugJson(problem.controls[term.firstControl]);
    }
    if (term.secondControl < problem.controls.size()) {
        out["second_control"] = controlDebugJson(problem.controls[term.secondControl]);
    }
    return out;
}

std::string controlDebugString(const AtlasSnapCandidateSet& control)
{
    std::ostringstream out;
    out << "id=" << control.id
        << " fiber=" << control.fiberPath.generic_string()
        << " source_index=" << control.sourceIndex
        << " manual=" << (control.manual ? "true" : "false")
        << " fixed=" << (control.fixed ? "true" : "false")
        << " eligible=" << (control.eligible ? "true" : "false")
        << " status=" << control.status
        << " reason=\"" << control.statusReason << '"'
        << " control_point=" << vecString(control.controlPoint)
        << " candidate_count=" << control.candidates.size()
        << " candidates=" << pointsString(control.candidates);
    return out.str();
}

bool predSnapPointCandidateReady(const AtlasPredSnapPoint& point)
{
    if (point.source == AtlasPredSnapSource::Manual && point.predSnapPoint) {
        return true;
    }
    if (point.source == AtlasPredSnapSource::Optimized && point.predSnapPoint) {
        return true;
    }
    if (!point.status.empty() &&
        point.status != "ready_single" &&
        point.status != "ready_inside" &&
        point.status != "ready_two_sided" &&
        point.status != "insufficient_candidates_outside" &&
        point.status != "optimized") {
        return false;
    }
    return !point.candidates.empty();
}

std::string predSnapPointStatus(const AtlasPredSnapPoint& point)
{
    if (point.status == "insufficient_candidates_outside") {
        return point.candidates.empty()
            ? "insufficient_candidates_none"
            : (point.candidates.size() >= 2 ? "ready_two_sided" : "ready_single");
    }
    if (!point.status.empty()) {
        return point.status;
    }
    if (point.source == AtlasPredSnapSource::Manual && point.predSnapPoint) {
        return "manual";
    }
    if (point.predSnapPoint && point.source == AtlasPredSnapSource::Optimized) {
        return "optimized";
    }
    if (predSnapPointCandidateReady(point)) {
        return point.candidates.size() >= 2 ? "ready_two_sided" : "ready_single";
    }
    return "insufficient_candidates_none";
}

std::string predSnapPointStatusReason(const AtlasPredSnapPoint& point)
{
    if (point.status == "insufficient_candidates_outside") {
        if (point.candidates.empty()) {
            return "legacy outside candidate record has zero usable candidates and remains insufficient under the current snap rule";
        }
        return "legacy outside candidate record has at least one usable candidate and is ready under the current snap rule";
    }
    if (!point.statusReason.empty()) {
        return point.statusReason;
    }
    if (point.source == AtlasPredSnapSource::Manual && point.predSnapPoint) {
        return "manual pred-snap point";
    }
    if (point.predSnapPoint && point.source == AtlasPredSnapSource::Optimized) {
        return "optimized pred-snap point";
    }
    if (predSnapPointCandidateReady(point)) {
        return "candidate set is ready for snap optimization";
    }
    return "candidate set is not sufficient for snap optimization";
}

AtlasPredSnapPoint* predSnapPointForControl(AtlasPredSnapSet& set,
                                            const cv::Vec3d& controlPoint)
{
    const std::string key = atlasPredSnapControlPointKey(controlPoint);
    auto it = std::find_if(set.points.begin(),
                           set.points.end(),
                           [&key](const AtlasPredSnapPoint& point) {
                               return atlasPredSnapControlPointKey(point.controlPoint) == key;
                           });
    return it == set.points.end() ? nullptr : &*it;
}

void propagateAtlasPredSnapStatuses(
    const Atlas& atlas,
    std::unordered_map<std::string, AtlasPredSnapSet>& setsByFiber)
{
    for (const auto& mapping : atlas.fibers) {
        const std::string fiberKey = atlasFiberPathKey(mapping.fiberPath);
        auto setIt = setsByFiber.find(fiberKey);
        if (setIt == setsByFiber.end()) {
            AtlasPredSnapSet set;
            set.fiberPath = mapping.fiberPath;
            setIt = setsByFiber.emplace(fiberKey, std::move(set)).first;
        }
        AtlasPredSnapSet& set = setIt->second;
        if (set.fiberPath.empty()) {
            set.fiberPath = mapping.fiberPath;
        }

        std::vector<const AtlasAnchor*> anchors;
        anchors.reserve(mapping.controlAnchors.size());
        for (const auto& anchor : mapping.controlAnchors) {
            anchors.push_back(&anchor);
        }
        std::sort(anchors.begin(),
                  anchors.end(),
                  [](const AtlasAnchor* a, const AtlasAnchor* b) {
                      return a->sourceIndex < b->sourceIndex;
                  });

        bool blocked = false;
        int blockerSourceIndex = -1;
        std::string blockerStatus;
        for (const AtlasAnchor* anchor : anchors) {
            AtlasPredSnapPoint* point = predSnapPointForControl(set, anchor->world);
            if (!point) {
                AtlasPredSnapPoint missing;
                missing.fiberPath = set.fiberPath;
                missing.sourceIndex = anchor->sourceIndex;
                missing.controlPoint = anchor->world;
                setPredSnapStatus(missing,
                                  "missing_snap_record",
                                  "no pred-snap attachment record exists for this atlas control");
                set.points.push_back(std::move(missing));
                point = &set.points.back();
            }
            point->fiberPath = set.fiberPath;
            point->sourceIndex = anchor->sourceIndex;

            if (blocked && point->source != AtlasPredSnapSource::Manual) {
                point->predSnapPoint.reset();
                point->selectedCandidateIndex.reset();
                point->source = AtlasPredSnapSource::Auto;
                std::ostringstream reason;
                reason << "previous control in this fiber chain has status "
                       << blockerStatus
                       << " at source_index " << blockerSourceIndex;
                setPredSnapStatus(*point, "blocked_by_previous_issue", reason.str());
                continue;
            }

            if (point->source == AtlasPredSnapSource::Manual && point->predSnapPoint) {
                setPredSnapStatus(*point, "manual", "manual pred-snap point");
                blocked = false;
                blockerSourceIndex = -1;
                blockerStatus.clear();
                continue;
            }

            if (!predSnapPointCandidateReady(*point)) {
                if (point->status.empty() ||
                    point->status == "insufficient_candidates_outside") {
                    setPredSnapStatus(*point,
                                      predSnapPointStatus(*point),
                                      predSnapPointStatusReason(*point));
                }
                point->predSnapPoint.reset();
                point->selectedCandidateIndex.reset();
                point->source = AtlasPredSnapSource::Auto;
                blocked = true;
                blockerSourceIndex = anchor->sourceIndex;
                blockerStatus = point->status;
                continue;
            }

            if (point->status.empty() ||
                point->status == "insufficient_candidates_outside") {
                setPredSnapStatus(*point,
                                  predSnapPointStatus(*point),
                                  predSnapPointStatusReason(*point));
            }
        }
    }
}

std::string termDebugString(
    const AtlasSnapPairTerm& term,
    const AtlasSnapOptimizationProblem& problem)
{
    std::ostringstream out;
    out << "term=" << term.id;
    if (term.firstControl < problem.controls.size()) {
        out << " first={" << controlDebugString(problem.controls[term.firstControl]) << '}';
    } else {
        out << " first_index=" << term.firstControl;
    }
    if (term.secondControl < problem.controls.size()) {
        out << " second={" << controlDebugString(problem.controls[term.secondControl]) << '}';
    } else {
        out << " second_index=" << term.secondControl;
    }
    return out.str();
}

nlohmann::json effectiveRankOptions(const AtlasSnapOptimizeOptions& options)
{
    nlohmann::json rankOptions = options.rankOptions;
    if (!rankOptions.is_object()) {
        throw std::runtime_error("atlas snap optimizer requires rankOptions to be a JSON object");
    }
    if (!rankOptions.contains("amgx_config")) {
        rankOptions["amgx_config"] = nullptr;
    }

    for (const char* key : {"adaptive_start_lambda", "solver_tolerance", "confidence_factor"}) {
        if (rankOptions.contains(key)) {
            throw std::runtime_error(
                std::string("atlas snap rank option is server-owned and must not be supplied: ") +
                key + "; rank_options=" + rankOptions.dump());
        }
    }

    const std::array<const char*, 4> requiredKeys{
        "threshold",
        "margin_base_voxels",
        "source_depth",
        "amgx_config",
    };
    for (const char* key : requiredKeys) {
        if (!rankOptions.contains(key)) {
            throw std::runtime_error(
                std::string("atlas snap rank cache requires explicit rank option: ") +
                key + "; rank_options=" + rankOptions.dump());
        }
        if (std::string_view(key) != "amgx_config" && rankOptions[key].is_null()) {
            throw std::runtime_error(
                std::string("atlas snap rank option must not be null: ") +
                key + "; rank_options=" + rankOptions.dump());
        }
    }

    const auto threshold = rankOptions["threshold"].get<int>();
    if (threshold < 0 || threshold > 255) {
        throw std::runtime_error("atlas snap rank option threshold must be in [0, 255]");
    }
    if (threshold != options.predDtThreshold) {
        throw std::runtime_error(
            "atlas snap candidate threshold and rank threshold differ; cache would be ambiguous");
    }
    const auto margin = rankOptions["margin_base_voxels"].get<int64_t>();
    if (margin < 0) {
        throw std::runtime_error("atlas snap rank option margin_base_voxels must be non-negative");
    }
    const auto sourceDepth = rankOptions["source_depth"].get<int64_t>();
    if (sourceDepth < 0) {
        throw std::runtime_error("atlas snap rank option source_depth must be non-negative");
    }
    if (!rankOptions["amgx_config"].is_null() && !rankOptions["amgx_config"].is_string()) {
        throw std::runtime_error("atlas snap rank option amgx_config must be a path string or null");
    }
    return rankOptions;
}

double matrixValueForAssignment(const AtlasSnapPairMatrix& matrix, size_t a, size_t b)
{
    if (a >= matrix.normalizedValues.size() ||
        b >= matrix.normalizedValues[a].size()) {
        return 0.0;
    }
    const double value = matrix.normalizedValues[a][b];
    return std::isfinite(value) ? value : 0.0;
}

bool matrixHasNonZeroContribution(const AtlasSnapPairMatrix& matrix)
{
    for (const auto& row : matrix.normalizedValues) {
        for (double value : row) {
            if (std::isfinite(value) && std::abs(value) > 0.0) {
                return true;
            }
        }
    }
    return false;
}

double assignmentScore(const AtlasSnapOptimizationProblem& problem,
                       const std::unordered_map<std::string, const AtlasSnapPairMatrix*>& matricesById,
                       const std::vector<size_t>& assignment)
{
    double score = 0.0;
    for (const auto& term : problem.terms) {
        const auto matrixIt = matricesById.find(term.id);
        if (matrixIt == matricesById.end() || !matrixIt->second) {
            continue;
        }
        if (term.firstControl >= assignment.size() ||
            term.secondControl >= assignment.size()) {
            continue;
        }
        score += matrixValueForAssignment(*matrixIt->second,
                                          assignment[term.firstControl],
                                          assignment[term.secondControl]);
    }
    return score;
}

} // namespace

std::string atlasSnapRankTermCacheKey(
    const fs::path& manifestPath,
    const nlohmann::json& rankOptions,
    const std::vector<cv::Vec3d>& sideA,
    const std::vector<cv::Vec3d>& sideB)
{
    nlohmann::json manifestIdentity = {
        {"path", manifestPath.lexically_normal().generic_string()},
    };
    std::error_code ec;
    if (fs::is_regular_file(manifestPath, ec)) {
        manifestIdentity["size"] = static_cast<uint64_t>(fs::file_size(manifestPath, ec));
        ec.clear();
        manifestIdentity["mtime"] =
            fs::last_write_time(manifestPath, ec).time_since_epoch().count();
    }
    nlohmann::json key = {
        {"schema", 2},
        {"rank_algorithm", "laplace_rank_fixed_lambda_search_v1"},
        {"manifest", manifestIdentity},
        {"options", rankOptions.is_null() ? nlohmann::json::object() : rankOptions},
        {"side_a", pointsJson(sideA)},
        {"side_b", pointsJson(sideB)},
    };
    return hex64(fnv1a64(key.dump()));
}

AtlasSnapOptimizationProblem buildAtlasSnapOptimizationProblem(
    const Atlas& atlas,
    const std::unordered_map<std::string, AtlasPredSnapSet>& predSnapSets)
{
    AtlasSnapOptimizationProblem problem;
    std::unordered_set<std::string> controlIds;
    for (const auto& mapping : atlas.fibers) {
        const std::string fiberKey = atlasFiberPathKey(mapping.fiberPath);
        const auto setIt = predSnapSets.find(fiberKey);
        if (setIt == predSnapSets.end()) {
            continue;
        }
        std::unordered_map<std::string, const AtlasPredSnapPoint*> snapByControlKey;
        for (const auto& point : setIt->second.points) {
            snapByControlKey.emplace(atlasPredSnapControlPointKey(point.controlPoint), &point);
        }
        for (const auto& anchor : mapping.controlAnchors) {
            const auto snapIt = snapByControlKey.find(atlasPredSnapControlPointKey(anchor.world));
            if (snapIt == snapByControlKey.end() || !snapIt->second) {
                continue;
            }
            const AtlasPredSnapPoint& point = *snapIt->second;
            AtlasSnapCandidateSet control;
            control.id = controlIdFor(mapping.fiberPath, anchor.sourceIndex);
            if (!controlIds.insert(control.id).second) {
                continue;
            }
            control.fiberPath = mapping.fiberPath;
            control.sourceIndex = anchor.sourceIndex;
            control.controlPoint = point.controlPoint;
            control.manual = point.source == AtlasPredSnapSource::Manual;
            control.status = predSnapPointStatus(point);
            control.statusReason = predSnapPointStatusReason(point);
            for (const auto& candidate : point.candidates) {
                if (finitePoint(candidate.point)) {
                    control.candidates.push_back(candidate.point);
                }
            }
            if (control.candidates.empty() &&
                point.predSnapPoint &&
                (point.source == AtlasPredSnapSource::Manual ||
                 point.source == AtlasPredSnapSource::Optimized)) {
                control.candidates.push_back(*point.predSnapPoint);
            }
            control.eligible =
                point.source == AtlasPredSnapSource::Manual ||
                (point.status != "blocked_by_previous_issue" &&
                 predSnapPointCandidateReady(point));
            if (!control.eligible) {
                continue;
            }
            control.fixed = control.manual || control.candidates.size() <= 1;
            problem.controls.push_back(std::move(control));
        }
    }

    std::unordered_set<std::string> termIds;
    auto addPairTerm = [&](size_t first, size_t second) {
        if (first == second) {
            return;
        }
        const std::string id = termIdFor(first, second);
        if (termIds.insert(id).second) {
            const auto ordered = std::minmax(first, second);
            problem.terms.push_back({id, ordered.first, ordered.second});
        }
    };

    for (const auto& mapping : atlas.fibers) {
        std::vector<int> indices;
        indices.reserve(mapping.controlAnchors.size());
        for (const auto& anchor : mapping.controlAnchors) {
            indices.push_back(anchor.sourceIndex);
        }
        std::sort(indices.begin(), indices.end());
        indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
        for (size_t i = 1; i < indices.size(); ++i) {
            const auto first = controlIndexForEndpoint(problem, mapping.fiberPath, indices[i - 1]);
            const auto second = controlIndexForEndpoint(problem, mapping.fiberPath, indices[i]);
            if (first && second) {
                addPairTerm(*first, *second);
            }
        }
    }

    for (size_t linkIndex = 0; linkIndex < atlas.links.size(); ++linkIndex) {
        const AtlasLink& link = atlas.links[linkIndex];
        const auto* firstMapping = mappingForPath(atlas, link.first.fiberPath);
        const auto* secondMapping = mappingForPath(atlas, link.second.fiberPath);
        if (!firstMapping || !secondMapping) {
            continue;
        }
        std::vector<size_t> controls;
        for (int sourceIndex : bracketingControlSourceIndices(*firstMapping,
                                                              link.first.sourceIndex)) {
            if (auto idx = controlIndexForEndpoint(problem,
                                                   firstMapping->fiberPath,
                                                   sourceIndex)) {
                controls.push_back(*idx);
            }
        }
        for (int sourceIndex : bracketingControlSourceIndices(*secondMapping,
                                                              link.second.sourceIndex)) {
            if (auto idx = controlIndexForEndpoint(problem,
                                                   secondMapping->fiberPath,
                                                   sourceIndex)) {
                controls.push_back(*idx);
            }
        }
        std::sort(controls.begin(), controls.end());
        controls.erase(std::unique(controls.begin(), controls.end()), controls.end());
        for (size_t i = 0; i < controls.size(); ++i) {
            for (size_t j = i + 1; j < controls.size(); ++j) {
                addPairTerm(controls[i], controls[j]);
            }
        }
        (void)linkIndex;
    }
    return problem;
}

AtlasSnapPairMatrix atlasSnapPairMatrixFromRankResult(
    const AtlasSnapPairTerm& term,
    size_t sideACount,
    size_t sideBCount,
    const nlohmann::json& result)
{
    AtlasSnapPairMatrix matrix;
    matrix.id = term.id;
    matrix.rawValues.assign(sideACount, std::vector<double>(sideBCount, 0.0));
    matrix.normalizedValues = matrix.rawValues;
    matrix.metadata = result;
    if (!result.is_object() || result.value("status", std::string{}) != "success") {
        std::string message = "rank result failed";
        std::string code;
        if (result.contains("error") && result["error"].is_object()) {
            code = result["error"].value("code", std::string{});
            message = result["error"].value("message", message);
        }
        const bool zeroContribution =
            code == "no_accepted_lambda" ||
            code == "cuda_failure" ||
            (code == "rank_failed" &&
             (message.find("pred_dt crop contains no passable voxels") != std::string::npos ||
              message.find("source/sink point is outside the pred_dt crop") != std::string::npos ||
              message.find("AMGX_") != std::string::npos ||
              message.find("Cuda failure") != std::string::npos ||
              message.find("CUDA failure") != std::string::npos));
        if (zeroContribution) {
            matrix.metadata["raw_matrix"] = matrix.rawValues;
            matrix.metadata["normalized_matrix"] = matrix.normalizedValues;
            matrix.metadata["zero_contribution"] = true;
            return matrix;
        }
        throw std::runtime_error("atlas snap pair " + term.id + ": " + message);
    }
    for (const auto& value : result.value("values", nlohmann::json::array())) {
        if (!value.is_object()) {
            continue;
        }
        const std::string solveSide = value.value("solve_side", std::string{});
        const size_t solveIndex = value.value("solve_index", 0);
        const size_t targetIndex = value.value("target_index", 0);
        const double score = value.value("value", 0.0);
        size_t a = 0;
        size_t b = 0;
        if (solveSide == "side_a") {
            a = solveIndex;
            b = targetIndex;
        } else {
            a = targetIndex;
            b = solveIndex;
        }
        if (a < sideACount && b < sideBCount && std::isfinite(score)) {
            matrix.rawValues[a][b] = score;
        }
    }
    double maxAbs = 0.0;
    for (const auto& row : matrix.rawValues) {
        for (double value : row) {
            maxAbs = std::max(maxAbs, std::abs(value));
        }
    }
    if (maxAbs > 0.0 && std::isfinite(maxAbs)) {
        for (size_t a = 0; a < matrix.rawValues.size(); ++a) {
            for (size_t b = 0; b < matrix.rawValues[a].size(); ++b) {
                matrix.normalizedValues[a][b] = matrix.rawValues[a][b] / maxAbs;
            }
        }
    }
    matrix.metadata["raw_matrix"] = matrix.rawValues;
    matrix.metadata["normalized_matrix"] = matrix.normalizedValues;
    return matrix;
}

AtlasSnapOptimizationResult optimizeAtlasSnapCandidates(
    const AtlasSnapOptimizationProblem& problem,
    const std::vector<AtlasSnapPairMatrix>& matrices,
    const AtlasSnapOptimizeOptions& options)
{
    AtlasSnapOptimizationResult result;
    result.selectedCandidateIndices.assign(problem.controls.size(), 0);
    std::unordered_map<std::string, const AtlasSnapPairMatrix*> matricesById;
    for (const auto& matrix : matrices) {
        matricesById[matrix.id] = &matrix;
    }

    std::vector<std::vector<size_t>> adjacency(problem.controls.size());
    for (const auto& term : problem.terms) {
        if (term.firstControl < adjacency.size() && term.secondControl < adjacency.size()) {
            adjacency[term.firstControl].push_back(term.secondControl);
            adjacency[term.secondControl].push_back(term.firstControl);
        }
    }

    std::vector<uint8_t> seen(problem.controls.size(), 0);
    for (size_t seed = 0; seed < problem.controls.size(); ++seed) {
        if (seen[seed]) {
            continue;
        }
        std::vector<size_t> component;
        std::queue<size_t> q;
        q.push(seed);
        seen[seed] = 1;
        while (!q.empty()) {
            const size_t cur = q.front();
            q.pop();
            component.push_back(cur);
            for (size_t next : adjacency[cur]) {
                if (!seen[next]) {
                    seen[next] = 1;
                    q.push(next);
                }
            }
        }

        std::vector<size_t> variables;
        size_t combinations = 1;
        bool exhaustive = true;
        for (size_t idx : component) {
            const size_t count = problem.controls[idx].candidates.size();
            if (!problem.controls[idx].fixed && count > 1) {
                variables.push_back(idx);
                if (combinations > options.exhaustiveAssignmentLimit / count) {
                    exhaustive = false;
                } else {
                    combinations *= count;
                }
            }
        }
        if (variables.empty()) {
            continue;
        }

        if (exhaustive) {
            std::vector<size_t> best = result.selectedCandidateIndices;
            double bestScore = assignmentScore(problem, matricesById, best);
            std::vector<size_t> cur = result.selectedCandidateIndices;
            std::function<void(size_t)> visit = [&](size_t varPos) {
                if (varPos == variables.size()) {
                    const double score = assignmentScore(problem, matricesById, cur);
                    if (score > bestScore + 1.0e-12) {
                        bestScore = score;
                        best = cur;
                    }
                    return;
                }
                const size_t controlIndex = variables[varPos];
                for (size_t cand = 0;
                     cand < problem.controls[controlIndex].candidates.size();
                     ++cand) {
                    cur[controlIndex] = cand;
                    visit(varPos + 1);
                }
            };
            visit(0);
            result.selectedCandidateIndices.swap(best);
        } else {
            for (int pass = 0; pass < 64; ++pass) {
                bool changed = false;
                for (size_t controlIndex : variables) {
                    size_t bestCand = result.selectedCandidateIndices[controlIndex];
                    double bestScore = assignmentScore(problem, matricesById,
                                                       result.selectedCandidateIndices);
                    for (size_t cand = 0;
                         cand < problem.controls[controlIndex].candidates.size();
                         ++cand) {
                        if (cand == bestCand) {
                            continue;
                        }
                        std::vector<size_t> trial = result.selectedCandidateIndices;
                        trial[controlIndex] = cand;
                        const double score = assignmentScore(problem, matricesById, trial);
                        if (score > bestScore + 1.0e-12) {
                            bestScore = score;
                            bestCand = cand;
                        }
                    }
                    if (bestCand != result.selectedCandidateIndices[controlIndex]) {
                        result.selectedCandidateIndices[controlIndex] = bestCand;
                        changed = true;
                    }
                }
                if (!changed) {
                    break;
                }
            }
        }
    }
    result.objective = assignmentScore(problem, matricesById, result.selectedCandidateIndices);
    return result;
}

struct AtlasSnapPreparedCandidatesState {
    fs::path atlasDir;
    fs::path manifestPath;
    fs::path cachePath;
    nlohmann::json rankOptions = nlohmann::json::object();
    nlohmann::json cache = nlohmann::json::object();
    Atlas atlas;
    std::unordered_map<std::string, AtlasPredSnapSet> setsByFiber;
    AtlasSnapOptimizationProblem problem;
    AtlasSnapOptimizeOptions options;
    AtlasSnapOptimizeReport report;
    std::vector<AtlasSnapPairMatrix> matrices;
    std::vector<size_t> missingTermIndices;
    std::vector<std::string> missingKeys;
    bool finished = false;
};

namespace {

void cachePreparedAtlasPredSnapRankResult(
    AtlasSnapPreparedCandidatesState& state,
    size_t index,
    const nlohmann::json& result)
{
    if (index >= state.missingTermIndices.size()) {
        std::cerr << "[atlas-snap] ignoring out-of-range partial rank result"
                  << " index=" << index
                  << " expected=" << state.missingTermIndices.size()
                  << std::endl;
        return;
    }
    const AtlasSnapPairTerm& term =
        state.problem.terms[state.missingTermIndices[index]];
    const auto& first = state.problem.controls[term.firstControl];
    const auto& second = state.problem.controls[term.secondControl];
    AtlasSnapPairMatrix matrix;
    try {
        matrix = atlasSnapPairMatrixFromRankResult(
            term, first.candidates.size(), second.candidates.size(), result);
    } catch (const std::exception& ex) {
        std::cerr << "[atlas-snap] not caching partial rank result "
                  << termDebugString(term, state.problem)
                  << " error=" << ex.what()
                  << std::endl;
        return;
    }
    state.cache["entries"][state.missingKeys[index]] = {
        {"schema", 1},
        {"source_job_id", term.id},
        {"manifest", state.manifestPath.lexically_normal().generic_string()},
        {"rank_options", state.rankOptions},
        {"side_a", pointsJson(first.candidates)},
        {"side_b", pointsJson(second.candidates)},
        {"rank_result", matrix.metadata},
    };
    try {
        writeJsonFile(state.cachePath, state.cache);
        std::cerr << "[atlas-snap] cached partial rank result"
                  << " index=" << index
                  << " id=" << term.id
                  << " entries=" << state.cache["entries"].size()
                  << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "[atlas-snap] could not write partial rank cache: "
                  << ex.what()
                  << std::endl;
    }
}

} // namespace

AtlasSnapPreparedCandidates prepareAtlasPredSnapCandidates(
    const fs::path& atlasDir,
    const fs::path& volpkgRootIn,
    const fs::path& manifestPath,
    const vc::lasagna::LasagnaNormalSampler& sampler,
    const AtlasSnapOptimizeOptions& options)
{
    AtlasSnapPreparedCandidates prepared;
    prepared.state = std::make_shared<AtlasSnapPreparedCandidatesState>();
    AtlasSnapPreparedCandidatesState& state = *prepared.state;
    state.atlasDir = atlasDir;
    state.manifestPath = manifestPath;
    state.options = options;
    const fs::path volpkgRoot = volpkgRootIn.empty()
        ? inferVolpkgRootFromAtlasDir(atlasDir)
        : volpkgRootIn;
    std::cerr << "[atlas-snap] loading atlas=" << atlasDir.string()
              << " volpkg_root=" << volpkgRoot.string()
              << std::endl;
    Atlas atlas = Atlas::load(atlasDir, volpkgRoot);
    std::cerr << "[atlas-snap] loaded atlas"
              << " fibers=" << atlas.fibers.size()
              << " links=" << atlas.links.size()
              << std::endl;
    std::cerr << "[atlas-snap] loading base mesh="
              << (atlasDir / atlas.metadata.baseMeshPath).string()
              << std::endl;
    QuadSurface baseSurface(atlasDir / atlas.metadata.baseMeshPath);
    const nlohmann::json rankOptions = effectiveRankOptions(options);
    state.rankOptions = rankOptions;
    const double predDtThreshold = rankOptions.value("threshold", options.predDtThreshold);
    std::cerr << "[atlas-snap] pred_dt threshold=" << predDtThreshold
              << " rank_options=" << rankOptions.dump()
              << std::endl;

    AtlasPredSnapSampling sampling;
    sampling.sampleNormal = [&sampler](const cv::Vec3d& point) {
        return sampler.sampleNormal(point);
    };
    sampling.samplePredDt = [&sampler](const cv::Vec3d& point) {
        return sampler.samplePredDt(point);
    };
    sampling.windingDistance = [&sampler](const cv::Vec3d& a,
                                          const cv::Vec3d& b,
                                          double stepVx) {
        return sampler.windingDistance(a, b, stepVx);
    };
    sampling.predDtThreshold = predDtThreshold;
    const auto predDtSpacing = sampler.predDtSpacing();
    if (!predDtSpacing || !std::isfinite(*predDtSpacing) || *predDtSpacing <= 0.0) {
        throw std::runtime_error("Lasagna pred_dt channel has no valid spacing");
    }
    sampling.predDtStepVx = 0.5 * *predDtSpacing;

    for (const auto& mapping : atlas.fibers) {
        const fs::path fiberPath = resolveAtlasRelativePath(atlasDir, volpkgRoot, mapping.fiberPath);
        std::cerr << "[atlas-snap] generating candidates fiber="
                  << mapping.fiberPath.generic_string()
                  << " controls=" << mapping.controlAnchors.size()
                  << std::endl;
        const FiberInput input = loadSourceFiberInput(fiberPath, mapping.fiberPath);
        AtlasPredSnapSet set = ensureAtlasPredSnapSet(atlasDir,
                                                      input,
                                                      mapping,
                                                      baseSurface,
                                                      sampling);
        std::cerr << "[atlas-snap] generated candidates fiber="
                  << mapping.fiberPath.generic_string()
                  << " points=" << set.points.size()
                  << std::endl;
        state.setsByFiber[atlasFiberPathKey(mapping.fiberPath)] = std::move(set);
    }

    propagateAtlasPredSnapStatuses(atlas, state.setsByFiber);

    state.atlas = std::move(atlas);
    state.problem =
        buildAtlasSnapOptimizationProblem(state.atlas, state.setsByFiber);

    state.cache = {
        {"type", "vc3d_atlas_pred_snap_rank_cache"},
        {"version", 1},
        {"entries", nlohmann::json::object()},
    };
    state.cachePath = atlasPredSnapRankCachePath(atlasDir);
    if (fs::is_regular_file(state.cachePath)) {
        try {
            std::cerr << "[atlas-snap] reading rank cache="
                      << state.cachePath.string()
                      << std::endl;
            state.cache = readJsonFile(state.cachePath);
            if (!state.cache.is_object() ||
                state.cache.value("type", std::string{}) != "vc3d_atlas_pred_snap_rank_cache" ||
                state.cache.value("version", 0) != 1 ||
                !state.cache.contains("entries") ||
                !state.cache["entries"].is_object()) {
                std::cerr << "[atlas-snap] rank cache has unexpected schema; ignoring entries"
                          << std::endl;
                state.cache["entries"] = nlohmann::json::object();
            }
        } catch (const std::exception& ex) {
            std::cerr << "[atlas-snap] could not read rank cache; ignoring: "
                      << ex.what() << std::endl;
            state.cache["entries"] = nlohmann::json::object();
        } catch (...) {
            std::cerr << "[atlas-snap] could not read rank cache; ignoring unknown error"
                      << std::endl;
            state.cache["entries"] = nlohmann::json::object();
        }
    }

    AtlasSnapOptimizeReport& report = state.report;
    report.controls = state.problem.controls.size();
    report.links = state.atlas.links.size();
    for (const auto& control : state.problem.controls) {
        if (!control.fixed && control.candidates.size() > 1) {
            ++report.variableControls;
        }
        if (control.fixed) {
            ++report.fixedControls;
        }
        if (control.manual) {
            ++report.manualControls;
        }
        if (control.fixed && !control.manual && control.candidates.size() <= 1) {
            ++report.singletonControls;
        }
    }
    report.pairTerms = state.problem.terms.size();
    std::cerr << "[atlas-snap] built optimization problem"
              << " controls=" << report.controls
              << " variables=" << report.variableControls
              << " fixed=" << report.fixedControls
              << " manual=" << report.manualControls
              << " singleton_auto=" << report.singletonControls
              << " links=" << report.links
              << " pair_terms=" << report.pairTerms
              << std::endl;

    nlohmann::json jobs = nlohmann::json::array();
    for (size_t termIndex = 0; termIndex < state.problem.terms.size(); ++termIndex) {
        const auto& term = state.problem.terms[termIndex];
        const auto& first = state.problem.controls[term.firstControl];
        const auto& second = state.problem.controls[term.secondControl];
        if (first.candidates.empty() || second.candidates.empty()) {
            ++report.skippedPairTerms;
            continue;
        }
        const std::string key = atlasSnapRankTermCacheKey(
            manifestPath, rankOptions, first.candidates, second.candidates);
        const auto cacheIt = state.cache["entries"].find(key);
        if (cacheIt != state.cache["entries"].end() && cacheIt->is_object() &&
            cacheIt->contains("rank_result")) {
            try {
                AtlasSnapPairMatrix matrix = atlasSnapPairMatrixFromRankResult(
                    term, first.candidates.size(), second.candidates.size(),
                    (*cacheIt)["rank_result"]);
                if (matrix.metadata.value("zero_contribution", false)) {
                    ++report.zeroContributionTerms;
                } else {
                    ++report.successfulPairTerms;
                }
                state.matrices.push_back(std::move(matrix));
            } catch (const std::exception& ex) {
                std::ostringstream message;
                message << "cached " << ex.what() << "; "
                        << termDebugString(term, state.problem);
                throw std::runtime_error(message.str());
            }
            ++report.cacheHits;
            continue;
        }
        jobs.push_back({
            {"id", term.id},
            {"side_a", pointsJson(first.candidates)},
            {"side_b", pointsJson(second.candidates)},
            {"debug", termDebugJson(term, state.problem)},
        });
        state.missingTermIndices.push_back(termIndex);
        state.missingKeys.push_back(key);
    }
    report.rankJobsRequested = state.missingTermIndices.size();

    std::cerr << "[atlas-snap] rank cache summary"
              << " hits=" << report.cacheHits
              << " misses=" << state.missingTermIndices.size()
              << " skipped=" << report.skippedPairTerms
              << std::endl;

    prepared.rankRequest = {
        {"manifest", manifestPath.string()},
        {"jobs", jobs},
        {"options", rankOptions},
    };
    std::cerr << "[atlas-snap] prepared rank request jobs="
              << state.missingTermIndices.size()
              << " manifest=" << manifestPath.string()
              << " options=" << prepared.rankRequest["options"].dump()
              << std::endl;
    return prepared;
}

void cacheAtlasPredSnapRankResult(
    const AtlasSnapPreparedCandidates& prepared,
    size_t index,
    const nlohmann::json& result)
{
    if (!prepared.state) {
        throw std::runtime_error("atlas snap rank result cache received an empty prepared session");
    }
    cachePreparedAtlasPredSnapRankResult(*prepared.state, index, result);
}

AtlasSnapOptimizeReport finishAtlasPredSnapCandidates(
    const AtlasSnapPreparedCandidates& prepared,
    const nlohmann::json& rankResponse)
{
    if (!prepared.state) {
        throw std::runtime_error("atlas snap optimizer received an empty prepared session");
    }
    AtlasSnapPreparedCandidatesState& state = *prepared.state;
    if (state.finished) {
        throw std::runtime_error("atlas snap prepared session was already finished");
    }
    state.finished = true;

    AtlasSnapOptimizeReport& report = state.report;
    AtlasSnapOptimizationProblem& problem = state.problem;
    std::vector<AtlasSnapPairMatrix>& matrices = state.matrices;

    if (!state.missingTermIndices.empty()) {
        const auto& results = rankResponse.at("results");
        if (!results.is_array() || results.size() != state.missingTermIndices.size()) {
            throw std::runtime_error("atlas snap rank response has an unexpected result count");
        }
        std::cerr << "[atlas-snap] parsing rank response results="
                  << results.size()
                  << std::endl;
        for (size_t i = 0; i < state.missingTermIndices.size(); ++i) {
            const auto& term = problem.terms[state.missingTermIndices[i]];
            const auto& first = problem.controls[term.firstControl];
            const auto& second = problem.controls[term.secondControl];
            const auto& result = results.at(i);
            AtlasSnapPairMatrix matrix;
            try {
                matrix = atlasSnapPairMatrixFromRankResult(
                    term, first.candidates.size(), second.candidates.size(), result);
            } catch (const std::exception& ex) {
                std::cerr << "[atlas-snap] rank result failed "
                          << termDebugString(term, problem)
                          << " result=" << result.dump()
                          << std::endl;
                std::ostringstream message;
                message << ex.what() << "; " << termDebugString(term, problem);
                throw std::runtime_error(message.str());
            }
            if (matrix.metadata.value("zero_contribution", false)) {
                ++report.zeroContributionTerms;
            } else {
                ++report.successfulPairTerms;
            }
            state.cache["entries"][state.missingKeys[i]] = {
                {"schema", 1},
                {"source_job_id", term.id},
                {"manifest", state.manifestPath.lexically_normal().generic_string()},
                {"rank_options", state.rankOptions},
                {"side_a", pointsJson(first.candidates)},
                {"side_b", pointsJson(second.candidates)},
                {"rank_result", matrix.metadata},
            };
            matrices.push_back(std::move(matrix));
        }
        std::cerr << "[atlas-snap] writing rank cache="
                  << state.cachePath.string()
                  << " entries=" << state.cache["entries"].size()
                  << std::endl;
        writeJsonFile(state.cachePath, state.cache);
    }

    std::unordered_set<std::string> nonZeroMatrixIds;
    for (const auto& matrix : matrices) {
        if (matrixHasNonZeroContribution(matrix)) {
            nonZeroMatrixIds.insert(matrix.id);
        }
    }
    std::unordered_map<std::string, const AtlasSnapPairMatrix*> matricesByIdForStatus;
    for (const auto& matrix : matrices) {
        matricesByIdForStatus[matrix.id] = &matrix;
    }
    std::vector<uint8_t> scoredControls(problem.controls.size(), 0);
    for (const auto& term : problem.terms) {
        if (nonZeroMatrixIds.find(term.id) == nonZeroMatrixIds.end()) {
            continue;
        }
        if (term.firstControl < scoredControls.size()) {
            scoredControls[term.firstControl] = 1;
        }
        if (term.secondControl < scoredControls.size()) {
            scoredControls[term.secondControl] = 1;
        }
    }
    std::unordered_map<size_t, std::pair<std::string, std::string>> inactiveControlStatuses;
    for (size_t i = 0; i < problem.controls.size(); ++i) {
        const auto& control = problem.controls[i];
        if (!control.fixed && control.candidates.size() > 1 && !scoredControls[i]) {
            ++report.unscoredVariableControls;
            std::vector<std::string> failures;
            for (const auto& term : problem.terms) {
                if (term.firstControl != i && term.secondControl != i) {
                    continue;
                }
                const auto matrixIt = matricesByIdForStatus.find(term.id);
                if (matrixIt == matricesByIdForStatus.end() || !matrixIt->second) {
                    failures.push_back(term.id + ": no rank result");
                    continue;
                }
                const auto& metadata = matrixIt->second->metadata;
                if (metadata.contains("error") && metadata["error"].is_object()) {
                    const std::string code =
                        metadata["error"].value("code", std::string("rank_failed"));
                    const std::string message =
                        metadata["error"].value("message", std::string("rank failed"));
                    failures.push_back(term.id + ": " + code + ": " + message);
                } else if (metadata.value("zero_contribution", false)) {
                    failures.push_back(term.id + ": zero_contribution");
                }
            }
            std::string reason =
                "all diffusion/rank pair terms involving this control failed or contributed zero";
            if (!failures.empty()) {
                reason += "; ";
                for (size_t j = 0; j < failures.size(); ++j) {
                    if (j > 0) {
                        reason += " | ";
                    }
                    reason += failures[j];
                }
            }
            inactiveControlStatuses.emplace(
                i,
                std::make_pair(
                    std::string("diffusion_failed"),
                    std::move(reason)));
        }
    }

    for (const auto& mapping : state.atlas.fibers) {
        std::vector<int> indices;
        indices.reserve(mapping.controlAnchors.size());
        for (const auto& anchor : mapping.controlAnchors) {
            indices.push_back(anchor.sourceIndex);
        }
        std::sort(indices.begin(), indices.end());
        indices.erase(std::unique(indices.begin(), indices.end()), indices.end());

        bool blocked = false;
        int blockerSourceIndex = -1;
        std::string blockerStatus;
        for (int sourceIndex : indices) {
            const auto idx =
                controlIndexForEndpoint(problem, mapping.fiberPath, sourceIndex);
            if (!idx) {
                blocked = true;
                blockerSourceIndex = sourceIndex;
                blockerStatus = "not_eligible";
                continue;
            }
            const auto& control = problem.controls[*idx];
            if (blocked && !control.manual) {
                std::ostringstream reason;
                reason << "previous control in this fiber chain has status "
                       << blockerStatus
                       << " at source_index " << blockerSourceIndex;
                inactiveControlStatuses.emplace(
                    *idx,
                    std::make_pair(std::string("blocked_by_previous_issue"),
                                   reason.str()));
                continue;
            }
            const auto inactiveIt = inactiveControlStatuses.find(*idx);
            if (inactiveIt != inactiveControlStatuses.end()) {
                blocked = true;
                blockerSourceIndex = sourceIndex;
                blockerStatus = inactiveIt->second.first;
                continue;
            }
            if (control.manual) {
                blocked = false;
                blockerSourceIndex = -1;
                blockerStatus.clear();
            }
        }
    }

    if (!inactiveControlStatuses.empty()) {
        const size_t beforeTerms = problem.terms.size();
        problem.terms.erase(
            std::remove_if(problem.terms.begin(),
                           problem.terms.end(),
                           [&inactiveControlStatuses](const AtlasSnapPairTerm& term) {
                               return inactiveControlStatuses.find(term.firstControl) !=
                                          inactiveControlStatuses.end() ||
                                      inactiveControlStatuses.find(term.secondControl) !=
                                          inactiveControlStatuses.end();
                           }),
            problem.terms.end());
        report.skippedPairTerms += beforeTerms - problem.terms.size();
        matrices.erase(
            std::remove_if(matrices.begin(),
                           matrices.end(),
                           [&problem](const AtlasSnapPairMatrix& matrix) {
                               return std::none_of(problem.terms.begin(),
                                                   problem.terms.end(),
                                                   [&matrix](const AtlasSnapPairTerm& term) {
                                                       return term.id == matrix.id;
                                                   });
                           }),
            matrices.end());
    }

    std::cerr << "[atlas-snap] running discrete optimizer"
              << " matrices=" << matrices.size()
              << " successful_terms=" << report.successfulPairTerms
              << " zero_contribution_terms=" << report.zeroContributionTerms
              << " skipped_terms=" << report.skippedPairTerms
              << " unscored_variables=" << report.unscoredVariableControls
              << std::endl;
    const AtlasSnapOptimizationResult optimized =
        optimizeAtlasSnapCandidates(problem, matrices, state.options);
    report.objective = optimized.objective;
    std::cerr << "[atlas-snap] discrete optimizer finished"
              << " objective=" << report.objective
              << std::endl;

    std::unordered_map<std::string, AtlasPredSnapSet*> mutableSets;
    for (auto& [key, set] : state.setsByFiber) {
        mutableSets[key] = &set;
    }
    for (size_t controlIndex = 0; controlIndex < problem.controls.size(); ++controlIndex) {
        const auto& control = problem.controls[controlIndex];
        auto setIt = mutableSets.find(atlasFiberPathKey(control.fiberPath));
        if (setIt == mutableSets.end() || !setIt->second) {
            continue;
        }
        const std::string controlKey = atlasPredSnapControlPointKey(control.controlPoint);
        auto pointIt = std::find_if(setIt->second->points.begin(),
                                    setIt->second->points.end(),
                                    [&controlKey](const AtlasPredSnapPoint& point) {
                                        return atlasPredSnapControlPointKey(point.controlPoint) == controlKey;
                                    });
        if (pointIt == setIt->second->points.end()) {
            continue;
        }
        const auto inactiveIt = inactiveControlStatuses.find(controlIndex);
        if (inactiveIt != inactiveControlStatuses.end()) {
            pointIt->predSnapPoint.reset();
            pointIt->selectedCandidateIndex.reset();
            pointIt->source = AtlasPredSnapSource::Auto;
            pointIt->predDtValue.reset();
            pointIt->direction.reset();
            pointIt->weightedFirstHitWindingDistance.reset();
            setPredSnapStatus(*pointIt,
                              inactiveIt->second.first,
                              inactiveIt->second.second);
            continue;
        }
        if (control.candidates.empty()) {
            pointIt->predSnapPoint.reset();
            pointIt->selectedCandidateIndex.reset();
            pointIt->source = AtlasPredSnapSource::Auto;
            setPredSnapStatus(*pointIt,
                              control.status.empty() ? "insufficient_candidates_none"
                                                     : control.status,
                              control.statusReason.empty()
                                  ? "no usable snap candidates"
                                  : control.statusReason);
            continue;
        }
        const size_t selected = controlIndex < optimized.selectedCandidateIndices.size()
            ? std::min(optimized.selectedCandidateIndices[controlIndex],
                       control.candidates.size() - 1)
            : 0;
        pointIt->predSnapPoint = control.candidates[selected];
        pointIt->selectedCandidateIndex = static_cast<int>(selected);
        pointIt->source = control.manual
            ? AtlasPredSnapSource::Manual
            : AtlasPredSnapSource::Optimized;
        setPredSnapStatus(*pointIt,
                          control.manual ? "manual" : "optimized",
                          control.manual ? "manual pred-snap point"
                                         : "selected by atlas snap discrete optimizer");
        if (selected < pointIt->candidates.size()) {
            pointIt->predDtValue = pointIt->candidates[selected].predDtValue;
            pointIt->direction = pointIt->candidates[selected].direction;
            pointIt->weightedFirstHitWindingDistance =
                pointIt->candidates[selected].windingDistance;
        }
    }
    for (const auto& [key, set] : state.setsByFiber) {
        (void)key;
        std::cerr << "[atlas-snap] saving pred snap attachment="
                  << atlasPredSnapAttachmentPath(state.atlasDir, set.fiberPath).string()
                  << std::endl;
        saveAtlasPredSnapSet(atlasPredSnapAttachmentPath(state.atlasDir, set.fiberPath), set);
    }
    return report;
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
                                          const fs::path& volpkgRootIn,
                                          const fs::path& fiberPathRoot)
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
    exportData.atlas = Atlas::load(atlasDir, exportData.volpkgRoot, fiberPathRoot);
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
        const fs::path fiberPath = resolveAtlasFiberPath(
            atlasDir, exportData.volpkgRoot, fiberPathRoot, mapping.fiberPath);
        if (!fs::is_regular_file(fiberPath)) {
            throw std::runtime_error("Atlas mapping " + mappingFiles[i].string() +
                                     " references missing fiber path: " +
                                     mapping.fiberPath.generic_string());
        }
        const FiberInput sourceFiber = loadSourceFiberInput(fiberPath, mapping.fiberPath);
        validateMappingControlAnchorsAgainstFiber(mapping, sourceFiber, mappingFiles[i]);

        LasagnaAtlasObject object;
        object.id = mapping.fiberPath.generic_string();
        object.fiberPath = fiberPath;
        object.mappingPath = mappingFiles[i];
        object.predSnapAttachmentPath = atlasPredSnapAttachmentPath(atlasDir, mapping.fiberPath);
        object.fiberRelativePath = mapping.fiberPath;
        object.mappingRelativePath = fs::relative(mappingFiles[i], atlasDir).lexically_normal();
        if (fs::is_regular_file(object.predSnapAttachmentPath)) {
            object.predSnapAttachmentRelativePath =
                fs::relative(object.predSnapAttachmentPath, atlasDir).lexically_normal();
        }
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
        nlohmann::json mapEntry = {
            {"object_type", "line"},
            {"object_id", object.id},
            {"fiber_path", object.fiberRelativePath.generic_string()},
            {"mapping_path", object.mappingRelativePath.generic_string()},
            {"winding_offset", object.windingOffset},
        };
        if (!object.predSnapAttachmentRelativePath.empty()) {
            mapEntry["pred_snap_path"] = object.predSnapAttachmentRelativePath.generic_string();
        }
        maps.push_back(std::move(mapEntry));
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
    copy.meta = surface.meta;
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

std::optional<cv::Vec3d> atlasAnchorBaseNormal(const AtlasAnchor& anchor,
                                               const FiberMapping& fiber,
                                               const QuadSurface& baseSurface)
{
    return baseNormalForAnchor(anchor,
                               fiber,
                               baseSurface,
                               atlasBaseNormalOutwardSign(baseSurface));
}

const AtlasAnchor* nearestLineAnchorForPosition(const FiberMapping& mapping,
                                                double linePosition)
{
    if (!std::isfinite(linePosition)) {
        return nullptr;
    }
    const AtlasAnchor* best = nullptr;
    double bestDelta = std::numeric_limits<double>::infinity();
    for (const auto& anchor : mapping.lineAnchors) {
        const double delta = std::abs(static_cast<double>(anchor.sourceIndex) - linePosition);
        if (delta < bestDelta) {
            best = &anchor;
            bestDelta = delta;
        }
    }
    return best;
}

AtlasSignedWindingDisplay signedAtlasSearchWindingDisplay(
    double windingDistance,
    bool sourceFiberDisplaysAsH,
    double sourceLinePosition,
    double targetLinePosition,
    const cv::Vec3d& sourcePoint,
    const cv::Vec3d& targetPoint,
    const FiberMapping& sourceMapping,
    const FiberMapping& targetMapping,
    const QuadSurface& baseSurface)
{
    if (!std::isfinite(windingDistance)) {
        throw std::runtime_error("atlas search winding distance is not finite");
    }
    if (!std::isfinite(sourceLinePosition) || !std::isfinite(targetLinePosition)) {
        throw std::runtime_error("atlas search line position is not finite");
    }
    if (!finitePoint(sourcePoint) || !finitePoint(targetPoint)) {
        throw std::runtime_error("atlas search intersection point is not finite");
    }

    const FiberMapping& hMapping = sourceFiberDisplaysAsH ? sourceMapping : targetMapping;
    const double hLinePosition = sourceFiberDisplaysAsH ? sourceLinePosition : targetLinePosition;
    const cv::Vec3d hPoint = sourceFiberDisplaysAsH ? sourcePoint : targetPoint;
    const cv::Vec3d vPoint = sourceFiberDisplaysAsH ? targetPoint : sourcePoint;

    const AtlasAnchor* hAnchor = nearestLineAnchorForPosition(hMapping, hLinePosition);
    if (!hAnchor) {
        throw std::runtime_error("mapped H fiber has no line anchor for atlas search signing");
    }
    const auto hNormal = atlasAnchorBaseNormal(*hAnchor, hMapping, baseSurface);
    if (!hNormal || !validNormal(*hNormal)) {
        throw std::runtime_error("could not sample outward base normal for atlas search signing");
    }

    const double projection = (vPoint - hPoint).dot(*hNormal);
    if (!std::isfinite(projection)) {
        throw std::runtime_error("atlas search H/V normal projection is not finite");
    }

    AtlasSignedWindingDisplay display;
    display.sourceFiberIsH = sourceFiberDisplaysAsH;
    display.hAnchorSourceIndex = hAnchor->sourceIndex;
    display.hToVOutwardProjection = projection;
    display.signedWindingDistance =
        projection > 0.0 ? -std::abs(windingDistance) : std::abs(windingDistance);
    return display;
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

int atlasLinkWindingOffsetDelta(const AtlasLink& link,
                                int periodColumns,
                                int zeroWindingColumn)
{
    const int firstBaseWinding = atlasWindingForColumn(
        link.first.atlasU, periodColumns, zeroWindingColumn);
    const int secondBaseWinding = atlasWindingForColumn(
        link.second.atlasU, periodColumns, zeroWindingColumn);
    return link.desiredWindingDelta - (secondBaseWinding - firstBaseWinding);
}

std::vector<AtlasLayoutConflict> layoutAtlasObjects(Atlas& atlas, int periodColumns)
{
    if (periodColumns <= 0 || atlas.fibers.empty()) {
        return {};
    }

    std::unordered_map<std::string, size_t> fiberIndexByPath;
    fiberIndexByPath.reserve(atlas.fibers.size());
    for (size_t i = 0; i < atlas.fibers.size(); ++i) {
        fiberIndexByPath.emplace(atlasFiberPathKey(atlas.fibers[i].fiberPath), i);
        atlas.fibers[i].windingOffset = 0;
    }

    struct Edge {
        size_t to = 0;
        int delta = 0;
        std::string toKey;
    };
    std::vector<std::vector<Edge>> graph(atlas.fibers.size());
    for (const auto& link : atlas.links) {
        const std::string firstKey = atlasFiberPathKey(link.first.fiberPath);
        const std::string secondKey = atlasFiberPathKey(link.second.fiberPath);
        const auto firstIt = fiberIndexByPath.find(firstKey);
        const auto secondIt = fiberIndexByPath.find(secondKey);
        if (firstIt == fiberIndexByPath.end() || secondIt == fiberIndexByPath.end()) {
            continue;
        }
        const int secondMinusFirst = atlasLinkWindingOffsetDelta(
            link, periodColumns, atlas.metadata.zeroWindingColumn);
        graph[firstIt->second].push_back({secondIt->second, secondMinusFirst, secondKey});
        graph[secondIt->second].push_back({firstIt->second, -secondMinusFirst, firstKey});
    }

    for (auto& edges : graph) {
        std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
            if (a.toKey != b.toKey) return a.toKey < b.toKey;
            if (a.delta != b.delta) return a.delta < b.delta;
            return a.to < b.to;
        });
    }

    std::vector<bool> visited(atlas.fibers.size(), false);
    std::vector<AtlasLayoutConflict> conflicts;
    std::queue<size_t> pending;
    visited[0] = true;
    pending.push(0);
    while (!pending.empty()) {
        const size_t current = pending.front();
        pending.pop();
        for (const auto& edge : graph[current]) {
            const int candidateOffset = atlas.fibers[current].windingOffset + edge.delta;
            if (visited[edge.to]) {
                if (atlas.fibers[edge.to].windingOffset != candidateOffset) {
                    conflicts.push_back({
                        atlas.fibers[edge.to].fiberPath,
                        atlas.fibers[edge.to].windingOffset,
                        candidateOffset,
                    });
                    atlasDebug(
                        "layout conflict for " +
                        atlas.fibers[edge.to].fiberPath.generic_string() +
                        ": existing offset " +
                        std::to_string(atlas.fibers[edge.to].windingOffset) +
                        ", candidate offset " + std::to_string(candidateOffset));
                }
                continue;
            }
            atlas.fibers[edge.to].windingOffset = candidateOffset;
            visited[edge.to] = true;
            pending.push(edge.to);
        }
    }
    return conflicts;
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
    anchors[seedIndex] = anchorFromHit(seedIndex,
                                       validatedFiber.linePoints[static_cast<size_t>(seedIndex)],
                                       hitsByLinePoint[seedIndex].front());
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

    for (size_t controlIndex = 0; controlIndex < validatedFiber.controlLineIndices.size(); ++controlIndex) {
        const int lineIndex = validatedFiber.controlLineIndices[controlIndex];
        if (lineIndex < mappedFirst || lineIndex > mappedLast) {
            continue;
        }
        const auto& anchor = anchors[static_cast<size_t>(lineIndex)];
        if (anchor) {
            AtlasAnchor control = *anchor;
            control.world = validatedFiber.controlPoints[controlIndex];
            mapping.controlAnchors.push_back(control);
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
