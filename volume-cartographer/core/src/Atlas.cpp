#include "vc/atlas/Atlas.hpp"

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

std::optional<AtlasAnchor> chooseContinuationHit(int sourceIndex,
                                                 const std::vector<ProjectionHit>& hits,
                                                 const AtlasAnchor& previous,
                                                 const cv::Vec3d& previousLinePoint,
                                                 const cv::Vec3d& linePoint,
                                                 int columns,
                                                 double mismatchRatio)
{
    if (hits.empty() || columns <= 0) {
        return std::nullopt;
    }
    const double lineStep = distance(previousLinePoint, linePoint);
    double bestScore = std::numeric_limits<double>::infinity();
    std::optional<AtlasAnchor> best;
    for (const auto& hit : hits) {
        const int prevWinding = static_cast<int>(std::floor(previous.atlasU / columns));
        for (int winding = prevWinding - 1; winding <= prevWinding + 1; ++winding) {
            AtlasAnchor candidate = anchorFromHit(sourceIndex, hit);
            candidate.atlasU = hit.atlasU + static_cast<double>(winding * columns);
            const double du = candidate.atlasU - previous.atlasU;
            const double dv = candidate.atlasV - previous.atlasV;
            const double gridStep = std::sqrt(du * du + dv * dv);
            if (lineStep > kEpsilon && gridStep > kEpsilon) {
                const double ratio = std::max(gridStep, lineStep) / std::min(gridStep, lineStep);
                if (ratio > mismatchRatio) {
                    continue;
                }
            }
            if (gridStep < bestScore) {
                bestScore = gridStep;
                best = candidate;
            }
        }
    }
    return best;
}

} // namespace

void Atlas::save(const fs::path& atlasDir) const
{
    nlohmann::json metadataJson = {
        {"type", metadata.type},
        {"version", metadata.version},
        {"name", metadata.name},
        {"base_mesh_path", metadata.baseMeshPath.generic_string()},
        {"source_base_mesh_path", metadata.sourceBaseMeshPath.generic_string()},
        {"idx_rotation_columns", metadata.idxRotationColumns},
        {"seed_line_index", metadata.seedLineIndex},
        {"seed_atlas", nlohmann::json::array({metadata.seedAtlasU, metadata.seedAtlasV})},
    };
    writeJsonFile(atlasDir / "metadata.json", metadataJson);

    nlohmann::json linksJson;
    linksJson["links"] = nlohmann::json::array();
    for (const auto& link : links) {
        linksJson["links"].push_back(link);
    }
    writeJsonFile(atlasDir / "links.json", linksJson);

    for (const auto& fiber : fibers) {
        nlohmann::json root;
        root["type"] = "vc3d_atlas_fiber_mapping";
        root["version"] = 1;
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
    if (atlas.metadata.type != "vc3d_atlas" || atlas.metadata.version != 1) {
        throw std::runtime_error("unsupported atlas metadata in " + atlasDir.string());
    }
    atlas.metadata.name = metadata.value("name", atlasDir.filename().string());
    atlas.metadata.baseMeshPath = metadata.value("base_mesh_path", std::string{});
    atlas.metadata.sourceBaseMeshPath = metadata.value("source_base_mesh_path", std::string{});
    atlas.metadata.idxRotationColumns = metadata.value("idx_rotation_columns", 0);
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
                atlas.links.push_back(link.get<std::string>());
            }
        }
    }

    const fs::path mappingsDir = atlasDir / "mappings" / "fibers";
    if (fs::is_directory(mappingsDir)) {
        for (const auto& entry : fs::directory_iterator(mappingsDir)) {
            if (!entry.is_regular_file() || entry.path().extension() != ".json") {
                continue;
            }
            const auto root = readJsonFile(entry.path());
            FiberMapping mapping;
            mapping.fiberPath = root.value("fiber_path", std::string{});
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

int computeIdxRotationColumns(const QuadSurface& surface)
{
    const auto* points = surface.rawPointsPtr();
    if (!points || points->cols <= 0) {
        return 0;
    }

    int bestCol = 0;
    double bestAverageY = std::numeric_limits<double>::infinity();
    for (int col = 0; col < points->cols; ++col) {
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

std::shared_ptr<QuadSurface> idxRotatedSurface(const QuadSurface& surface,
                                               int rotationColumns)
{
    const auto* points = surface.rawPointsPtr();
    if (!points) {
        throw std::runtime_error("base surface has no point grid");
    }
    const int cols = points->cols;
    const int rows = points->rows;
    const int shift = cols > 0 ? ((rotationColumns % cols) + cols) % cols : 0;
    cv::Mat_<cv::Vec3f> rotated(rows, cols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            rotated(row, col) = (*points)(row, (col + shift) % cols);
        }
    }

    auto out = std::make_shared<QuadSurface>(rotated, surface.scale());
    auto& mutableSurface = const_cast<QuadSurface&>(surface);
    for (const auto& name : mutableSurface.channelNames()) {
        cv::Mat channel = mutableSurface.channel(name);
        if (channel.empty() || channel.cols != cols || channel.rows != rows) {
            continue;
        }
        cv::Mat rotatedChannel(channel.rows, channel.cols, channel.type());
        for (int row = 0; row < channel.rows; ++row) {
            for (int col = 0; col < channel.cols; ++col) {
                channel.col((col + shift) % cols).row(row).copyTo(rotatedChannel.col(col).row(row));
            }
        }
        out->setChannel(name, rotatedChannel);
    }
    return out;
}

void saveIdxRotatedBaseMesh(const QuadSurface& surface,
                            int rotationColumns,
                            const fs::path& targetDir)
{
    auto rotated = idxRotatedSurface(surface, rotationColumns);
    rotated->save(targetDir, true);
}

AtlasCoveredSize mappedObjectCoveredAtlasSize(const Atlas& atlas)
{
    bool haveAnchor = false;
    double minU = std::numeric_limits<double>::infinity();
    double minV = std::numeric_limits<double>::infinity();
    double maxU = -std::numeric_limits<double>::infinity();
    double maxV = -std::numeric_limits<double>::infinity();

    auto includeAnchor = [&](const AtlasAnchor& anchor) {
        if (!std::isfinite(anchor.atlasU) || !std::isfinite(anchor.atlasV)) {
            return;
        }
        haveAnchor = true;
        minU = std::min(minU, anchor.atlasU);
        minV = std::min(minV, anchor.atlasV);
        maxU = std::max(maxU, anchor.atlasU);
        maxV = std::max(maxV, anchor.atlasV);
    };

    for (const auto& fiber : atlas.fibers) {
        for (const auto& anchor : fiber.lineAnchors) {
            includeAnchor(anchor);
        }
    }

    if (!haveAnchor) {
        return {};
    }

    AtlasCoveredSize size;
    size.width = std::max(1, static_cast<int>(std::ceil(maxU) - std::floor(minU) + 1.0));
    size.height = std::max(1, static_cast<int>(std::ceil(maxV) - std::floor(minV) + 1.0));
    size.valid = true;
    return size;
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
    auto includeAnchor = [&](const AtlasAnchor& anchor) {
        if (!std::isfinite(anchor.atlasU)) {
            return;
        }
        const int winding = static_cast<int>(std::floor(anchor.atlasU / static_cast<double>(baseColumns)));
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
            includeAnchor(anchor);
        }
    }

    if (!haveAnchor) {
        minWinding = static_cast<int>(std::floor(atlas.metadata.seedAtlasU / static_cast<double>(baseColumns)));
        maxWinding = minWinding;
    }

    range.leftmostWinding = minWinding;
    range.rightmostWinding = maxWinding;
    range.unwrapCount = std::max(1, maxWinding - minWinding + 1);
    range.atlasUOffset = static_cast<double>(minWinding * baseColumns);
    range.hasMappedObjects = haveAnchor;
    return range;
}

cv::Vec2f atlasGridToSurfaceCoords(double atlasU,
                                   double atlasV,
                                   const QuadSurface& displaySurface,
                                   double atlasUOffset)
{
    const cv::Vec3f center = displaySurface.center();
    const cv::Vec2f scale = displaySurface.scale();
    return {
        static_cast<float>((atlasU - atlasUOffset) - static_cast<double>(center[0] * scale[0])),
        static_cast<float>(atlasV - static_cast<double>(center[1] * scale[1])),
    };
}

std::shared_ptr<QuadSurface> repeatedAtlasDisplaySurface(const QuadSurface& baseSurface,
                                                        int unwrapCount)
{
    const auto* points = baseSurface.rawPointsPtr();
    if (!points) {
        throw std::runtime_error("base surface has no point grid");
    }
    if (unwrapCount <= 1) {
        return std::make_shared<QuadSurface>(*points, baseSurface.scale());
    }

    const int rows = points->rows;
    const int cols = points->cols;
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("base surface has no valid grid");
    }

    cv::Mat_<cv::Vec3f> repeated(rows, cols * unwrapCount);
    for (int unwrap = 0; unwrap < unwrapCount; ++unwrap) {
        const cv::Rect dst(unwrap * cols, 0, cols, rows);
        points->copyTo(repeated(dst));
    }

    auto out = std::make_shared<QuadSurface>(repeated, baseSurface.scale());
    auto& mutableSurface = const_cast<QuadSurface&>(baseSurface);
    for (const auto& name : mutableSurface.channelNames()) {
        cv::Mat channel = mutableSurface.channel(name);
        if (channel.empty() || channel.cols != cols || channel.rows != rows) {
            continue;
        }
        cv::Mat repeatedChannel(rows, cols * unwrapCount, channel.type());
        for (int unwrap = 0; unwrap < unwrapCount; ++unwrap) {
            channel.copyTo(repeatedChannel(cv::Rect(unwrap * cols, 0, cols, rows)));
        }
        out->setChannel(name, repeatedChannel);
    }
    return out;
}

FiberMapping mapFiberToBaseSurface(const FiberInput& fiber,
                                   const QuadSurface& baseSurface,
                                   SurfacePatchIndex& baseIndex,
                                   const vc::lasagna::NormalSampler& normalSampler,
                                   const LineMappingOptions& options)
{
    if (fiber.linePoints.empty()) {
        throw std::runtime_error("fiber has no line points");
    }
    const auto* points = baseSurface.rawPointsPtr();
    if (!points || points->cols <= 0) {
        throw std::runtime_error("base surface has no valid grid");
    }

    SurfacePatchIndex::SurfacePtr baseSurfacePtr(const_cast<QuadSurface*>(&baseSurface), [](QuadSurface*) {});
    const std::vector<SurfaceCandidate> baseCandidates = {{
        "base",
        {},
        baseSurfacePtr,
    }};

    const int seedIndex = seedLineIndexForFiber(fiber);
    atlasDebug("map fiber line_points=" + std::to_string(fiber.linePoints.size()) +
               " control_points=" + std::to_string(fiber.controlPoints.size()) +
               " seed_index=" + std::to_string(seedIndex));
    std::vector<std::vector<ProjectionHit>> hitsByLinePoint(fiber.linePoints.size());
    for (size_t i = 0; i < fiber.linePoints.size(); ++i) {
        const auto sample = normalSampler.sampleNormal(fiber.linePoints[i]);
        if (!sample.valid || !validNormal(sample.normal)) {
            atlasDebug("line_point[" + std::to_string(i) + "] invalid_normal point=" +
                       vecString(fiber.linePoints[i]));
            if (static_cast<int>(i) == seedIndex) {
                throw std::runtime_error("No valid normal at atlas seed point");
            }
            continue;
        }
        hitsByLinePoint[i] = projectPointToSurfacesAdaptive(
            fiber.linePoints[i], sample.normal, baseCandidates, baseIndex, options.rayHalfLength);
        if (hitsByLinePoint[i].empty()) {
            atlasDebug("line_point[" + std::to_string(i) + "] no_hits point=" +
                       vecString(fiber.linePoints[i]) + " normal=" + vecString(sample.normal));
        }
    }

    if (hitsByLinePoint[seedIndex].empty()) {
        throw std::runtime_error("failed to project atlas seed point onto the base shell");
    }

    std::vector<std::optional<AtlasAnchor>> anchors(fiber.linePoints.size());
    anchors[seedIndex] = anchorFromHit(seedIndex, hitsByLinePoint[seedIndex].front());
    atlasDebug("line_point[" + std::to_string(seedIndex) + "] chosen_anchor u=" +
               std::to_string(anchors[seedIndex]->atlasU) + " v=" +
               std::to_string(anchors[seedIndex]->atlasV));

    for (int i = seedIndex + 1; i < static_cast<int>(fiber.linePoints.size()); ++i) {
        const auto chosen = chooseContinuationHit(i,
                                                  hitsByLinePoint[i],
                                                  *anchors[i - 1],
                                                  fiber.linePoints[i - 1],
                                                  fiber.linePoints[i],
                                                  points->cols,
                                                  options.mismatchRatio);
        if (!chosen) {
            atlasDebug("line_point[" + std::to_string(i) + "] continuation_rejection");
            break;
        }
        anchors[i] = *chosen;
        atlasDebug("line_point[" + std::to_string(i) + "] chosen_anchor u=" +
                   std::to_string(anchors[i]->atlasU) + " v=" +
                   std::to_string(anchors[i]->atlasV));
    }
    for (int i = seedIndex - 1; i >= 0; --i) {
        const auto chosen = chooseContinuationHit(i,
                                                  hitsByLinePoint[i],
                                                  *anchors[i + 1],
                                                  fiber.linePoints[i + 1],
                                                  fiber.linePoints[i],
                                                  points->cols,
                                                  options.mismatchRatio);
        if (!chosen) {
            atlasDebug("line_point[" + std::to_string(i) + "] continuation_rejection");
            break;
        }
        anchors[i] = *chosen;
        atlasDebug("line_point[" + std::to_string(i) + "] chosen_anchor u=" +
                   std::to_string(anchors[i]->atlasU) + " v=" +
                   std::to_string(anchors[i]->atlasV));
    }

    FiberMapping mapping;
    mapping.fiberPath = fiber.fiberPath;
    for (const auto& anchor : anchors) {
        if (anchor) {
            mapping.lineAnchors.push_back(*anchor);
        }
    }
    atlasDebug("final_line_anchor_count=" + std::to_string(mapping.lineAnchors.size()));
    if (mapping.lineAnchors.size() < 2) {
        throw std::runtime_error("incomplete atlas mapping: produced fewer than two line anchors");
    }
    return mapping;
}

Atlas createSingleFiberAtlas(const fs::path& volpkgRoot,
                             const std::string& atlasName,
                             const FiberInput& fiber,
                             const SurfaceCandidate& baseSurface,
                             int rotationColumns,
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
    atlas.metadata.idxRotationColumns = rotationColumns;
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
