#include "vc/atlas/AtlasConstraints.hpp"

#include "vc/atlas/FiberHvClassification.hpp"
#include "vc/atlas/FiberIntersections.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace fs = std::filesystem;

namespace vc::atlas {
namespace {

constexpr double kEpsilon = 1.0e-9;

double norm(const cv::Vec3d& v)
{
    return std::sqrt(std::max(0.0, v.dot(v)));
}

bool finitePoint(const cv::Vec3d& p)
{
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

cv::Vec3d pointFromJson(const nlohmann::json& value)
{
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error("fiber point must be a 3-element array");
    }
    return {value[0].get<double>(), value[1].get<double>(), value[2].get<double>()};
}

std::vector<cv::Vec3d> pointArrayFromJson(const nlohmann::json& root,
                                          const char* key,
                                          const fs::path& path)
{
    if (!root.contains(key) || !root.at(key).is_array()) {
        throw std::runtime_error(path.string() + " is missing array field " + key);
    }
    std::vector<cv::Vec3d> out;
    for (const auto& point : root.at(key)) {
        out.push_back(pointFromJson(point));
    }
    return out;
}

struct LoadedFiber {
    FiberInput input;
    std::string manualHvTag;
    FiberHvClassification hv;
    std::vector<double> cumulativeArclength;
};

LoadedFiber loadFiberForConstraints(const fs::path& path,
                                    const fs::path& relativePath)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open fiber JSON: " + path.string());
    }
    const nlohmann::json root = nlohmann::json::parse(in);
    LoadedFiber loaded;
    loaded.input.fiberPath = relativePath;
    loaded.input.controlPoints = pointArrayFromJson(root, "control_points", path);
    loaded.input.linePoints = pointArrayFromJson(root, "line_points", path);
    validateFiberInputControlPoints(loaded.input);
    loaded.hv = classifyFiberHv(loaded.input.controlPoints);
    if (root.contains("hv_classification") && root.at("hv_classification").is_object()) {
        const auto& hv = root.at("hv_classification");
        if (hv.contains("manual_tag") && hv.at("manual_tag").is_string()) {
            const auto tag = fiberHvTagFromString(hv.at("manual_tag").get<std::string>());
            if (tag != FiberHvTag::Unknown) {
                loaded.manualHvTag = fiberHvTagToString(tag);
            }
        }
    }
    loaded.cumulativeArclength.assign(loaded.input.linePoints.size(), 0.0);
    for (size_t i = 1; i < loaded.input.linePoints.size(); ++i) {
        const double step = norm(loaded.input.linePoints[i] - loaded.input.linePoints[i - 1]);
        loaded.cumulativeArclength[i] =
            loaded.cumulativeArclength[i - 1] + (std::isfinite(step) ? step : 0.0);
    }
    return loaded;
}

double sourceIndexAtArclength(const LoadedFiber& fiber, double arclength)
{
    if (fiber.cumulativeArclength.empty()) {
        return 0.0;
    }
    if (arclength <= fiber.cumulativeArclength.front()) {
        return 0.0;
    }
    if (arclength >= fiber.cumulativeArclength.back()) {
        return static_cast<double>(fiber.cumulativeArclength.size() - 1);
    }
    auto it = std::lower_bound(fiber.cumulativeArclength.begin(),
                               fiber.cumulativeArclength.end(),
                               arclength);
    const size_t hi = static_cast<size_t>(std::distance(fiber.cumulativeArclength.begin(), it));
    const size_t lo = hi == 0 ? 0 : hi - 1;
    const double span = std::max(kEpsilon, fiber.cumulativeArclength[hi] - fiber.cumulativeArclength[lo]);
    const double t = std::clamp((arclength - fiber.cumulativeArclength[lo]) / span, 0.0, 1.0);
    return static_cast<double>(lo) * (1.0 - t) + static_cast<double>(hi) * t;
}

struct ExportPoint {
    size_t id = 0;
    size_t fiber = 0;
    double sourcePosition = 0.0;
    cv::Vec3d world{0.0, 0.0, 0.0};
    double atlasU = 0.0;
    double atlasV = 0.0;
    double continuousWinding = 0.0;
    std::string fiberDir;
    std::string stableKey;
};

struct GraphEdge {
    size_t a = 0;
    size_t b = 0;
    double credit = 0.0;
    bool covered = false;
    bool positive = false;
};

struct Graph {
    std::vector<ExportPoint> points;
    std::vector<GraphEdge> edges;
    std::vector<std::vector<size_t>> adjacency;
    int periodColumns = 0;
};

std::string pointKey(size_t fiber, double sourcePosition)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << fiber << ':' << std::fixed << std::setprecision(6) << sourcePosition;
    return out.str();
}

double continuousWindingForAtlasU(double actualU, const AtlasMetadata& metadata, int periodColumns)
{
    if (periodColumns <= 0) {
        return actualU;
    }
    return (actualU - static_cast<double>(metadata.zeroWindingColumn)) /
           static_cast<double>(periodColumns);
}

const AtlasAnchor* nearestControlAnchor(const FiberMapping& mapping, double sourcePosition)
{
    if (mapping.controlAnchors.empty() || !std::isfinite(sourcePosition)) {
        return nullptr;
    }
    const AtlasAnchor* best = nullptr;
    double bestDelta = std::numeric_limits<double>::infinity();
    for (const auto& anchor : mapping.controlAnchors) {
        const double delta = std::abs(static_cast<double>(anchor.sourceIndex) - sourcePosition);
        if (delta < bestDelta ||
            (delta == bestDelta && best && anchor.sourceIndex < best->sourceIndex)) {
            best = &anchor;
            bestDelta = delta;
        }
    }
    return best;
}

std::optional<double> nearestControlSourcePosition(const FiberMapping& mapping,
                                                   double sourcePosition)
{
    const AtlasAnchor* anchor = nearestControlAnchor(mapping, sourcePosition);
    if (!anchor) {
        return std::nullopt;
    }
    return static_cast<double>(anchor->sourceIndex);
}

std::optional<ExportPoint> interpolateMappedPoint(size_t fiberIndex,
                                                  const FiberMapping& mapping,
                                                  double sourcePosition,
                                                  const std::string& fiberDir,
                                                  const AtlasMetadata& metadata,
                                                  int periodColumns)
{
    const AtlasAnchor* anchor = nearestControlAnchor(mapping, sourcePosition);
    if (!anchor) {
        return std::nullopt;
    }

    ExportPoint p;
    p.fiber = fiberIndex;
    p.sourcePosition = static_cast<double>(anchor->sourceIndex);
    p.world = anchor->world;
    p.atlasU = actualAtlasU(*anchor, mapping, periodColumns);
    p.atlasV = anchor->atlasV;
    p.continuousWinding = continuousWindingForAtlasU(p.atlasU, metadata, periodColumns);
    p.fiberDir = fiberDir;
    p.stableKey = pointKey(fiberIndex, p.sourcePosition);
    if (!finitePoint(p.world) || !std::isfinite(p.continuousWinding)) {
        return std::nullopt;
    }
    return p;
}

void addUniquePosition(std::map<double, bool>& positions, double pos)
{
    if (std::isfinite(pos)) {
        positions[pos] = true;
    }
}

struct TemporaryLink {
    size_t firstFiber = 0;
    double firstPosition = 0.0;
    size_t secondFiber = 0;
    double secondPosition = 0.0;
};

std::vector<TemporaryLink> temporaryCycleClosingLinks(
    const LasagnaAtlasExport& exportData,
    const QuadSurface& baseSurface,
    const vc::lasagna::LasagnaNormalSampler* sampler,
    const std::vector<LoadedFiber>& fibers,
    const AtlasConstraintExportOptions& options,
    const std::unordered_map<std::string, size_t>& fiberIndexByKey)
{
    if (!options.closeCycles || !sampler || fibers.size() < 2) {
        return {};
    }

    std::vector<FiberPolyline> polylines;
    std::vector<uint64_t> ids;
    polylines.reserve(fibers.size());
    ids.reserve(fibers.size());
    for (size_t i = 0; i < fibers.size(); ++i) {
        FiberPolyline polyline;
        polyline.id = static_cast<uint64_t>(i + 1);
        polyline.generation = 1;
        polyline.controlPoints = fibers[i].input.controlPoints;
        for (const auto& p : fibers[i].input.linePoints) {
            polyline.points.push_back({p, std::nullopt});
        }
        ids.push_back(polyline.id);
        polylines.push_back(std::move(polyline));
    }

    FiberIntersectionBroadPhaseOptions broad;
    broad.maxDistance = options.intersectionMaxDistance;
    broad.maxSampleSpacing = options.intersectionMaxSampleSpacing;
    broad.seedStride = options.intersectionSeedStride;
    broad.clusterArclength = options.intersectionClusterArclength;
    FiberIntersectionCeresOptions ceres;
    ceres.maxIterations = options.intersectionMaxIterations;
    ceres.deduplicateArclength = options.intersectionDeduplicateArclength;

    FiberIntersectionCache cache;
    const auto results = searchFiberIntersections(
        polylines,
        ids,
        ids,
        &cache,
        broad,
        ceres,
        sampler);

    std::vector<TemporaryLink> links;
    for (const auto& result : results) {
        if (result.sourceFiberId == 0 || result.targetFiberId == 0 ||
            result.sourceFiberId > fibers.size() || result.targetFiberId > fibers.size()) {
            continue;
        }
        const size_t a = static_cast<size_t>(result.sourceFiberId - 1);
        const size_t b = static_cast<size_t>(result.targetFiberId - 1);
        const auto& sourceMapping = exportData.atlas.fibers[a];
        const auto& targetMapping = exportData.atlas.fibers[b];
        const bool sourceIsH = firstFiberDisplaysAsH(
            fibers[a].hv,
            fibers[a].manualHvTag,
            fibers[b].hv,
            fibers[b].manualHvTag,
            fiberIndexByKey.at(atlasFiberPathKey(sourceMapping.fiberPath)) <
                fiberIndexByKey.at(atlasFiberPathKey(targetMapping.fiberPath)));
        const double sourcePosition = sourceIndexAtArclength(fibers[a], result.sourceArclength);
        const double targetPosition = sourceIndexAtArclength(fibers[b], result.targetArclength);
        try {
            const auto display = signedAtlasSearchWindingDisplay(
                result.windingDistance,
                sourceIsH,
                sourcePosition,
                targetPosition,
                result.sourcePoint,
                result.targetPoint,
                sourceMapping,
                targetMapping,
                baseSurface);
            if (display.signedWindingDistance < options.closeMinSignedWinding ||
                display.signedWindingDistance > options.closeMaxSignedWinding) {
                continue;
            }
        } catch (const std::exception&) {
            continue;
        }

        const int period = atlasHorizontalPeriodColumns(baseSurface);
        const auto pa = interpolateMappedPoint(
            a,
            sourceMapping,
            sourcePosition,
            {},
            exportData.atlas.metadata,
            period);
        const auto pb = interpolateMappedPoint(
            b,
            targetMapping,
            targetPosition,
            {},
            exportData.atlas.metadata,
            period);
        if (!pa || !pb ||
            std::abs(pa->continuousWinding - pb->continuousWinding) >=
                options.closeAtlasWindingThreshold) {
            continue;
        }
        links.push_back({a, pa->sourcePosition, b, pb->sourcePosition});
    }
    std::sort(links.begin(), links.end(), [](const TemporaryLink& a, const TemporaryLink& b) {
        return std::tie(a.firstFiber, a.firstPosition, a.secondFiber, a.secondPosition) <
               std::tie(b.firstFiber, b.firstPosition, b.secondFiber, b.secondPosition);
    });
    return links;
}

size_t addGraphPoint(Graph& graph,
                     std::unordered_map<std::string, size_t>& idByKey,
                     ExportPoint point)
{
    const auto key = point.stableKey;
    const auto it = idByKey.find(key);
    if (it != idByKey.end()) {
        return it->second;
    }
    point.id = graph.points.size();
    idByKey[key] = point.id;
    graph.points.push_back(std::move(point));
    graph.adjacency.emplace_back();
    return graph.points.back().id;
}

void addGraphEdge(Graph& graph, size_t a, size_t b, double credit, bool positive)
{
    if (a == b) {
        return;
    }
    GraphEdge edge;
    edge.a = a;
    edge.b = b;
    edge.credit = std::max(0.0, credit);
    edge.positive = positive && edge.credit > kEpsilon;
    const size_t id = graph.edges.size();
    graph.edges.push_back(edge);
    graph.adjacency[a].push_back(id);
    graph.adjacency[b].push_back(id);
}

Graph buildGraph(const LasagnaAtlasExport& exportData,
                 const QuadSurface& baseSurface,
                 const std::vector<LoadedFiber>& fibers,
                 const std::vector<TemporaryLink>& temporaryLinks,
                 const AtlasConstraintExportOptions& options,
                 AtlasConstraintExportReport& report)
{
    (void)options;
    const int period = atlasHorizontalPeriodColumns(baseSurface);
    std::vector<std::map<double, bool>> fiberPositions(fibers.size());
    for (size_t i = 0; i < exportData.atlas.fibers.size(); ++i) {
        const auto& mapping = exportData.atlas.fibers[i];
        for (const auto& anchor : mapping.controlAnchors) {
            addUniquePosition(fiberPositions[i], static_cast<double>(anchor.sourceIndex));
        }
    }

    auto addLinkEndpoint = [&](const AtlasLinkEndpoint& endpoint) {
        const auto it = std::find_if(
            exportData.atlas.fibers.begin(),
            exportData.atlas.fibers.end(),
            [&](const FiberMapping& mapping) {
                return atlasFiberPathKey(mapping.fiberPath) == atlasFiberPathKey(endpoint.fiberPath);
            });
        if (it == exportData.atlas.fibers.end()) {
            return;
        }
        const size_t idx = static_cast<size_t>(std::distance(exportData.atlas.fibers.begin(), it));
        const auto snapped = nearestControlSourcePosition(
            exportData.atlas.fibers[idx],
            static_cast<double>(endpoint.sourceIndex));
        if (snapped) {
            addUniquePosition(fiberPositions[idx], *snapped);
        }
    };
    for (const auto& link : exportData.atlas.links) {
        addLinkEndpoint(link.first);
        addLinkEndpoint(link.second);
    }
    for (const auto& link : temporaryLinks) {
        addUniquePosition(fiberPositions[link.firstFiber], link.firstPosition);
        addUniquePosition(fiberPositions[link.secondFiber], link.secondPosition);
    }

    Graph graph;
    graph.periodColumns = period;
    std::unordered_map<std::string, size_t> idByKey;
    std::vector<std::vector<size_t>> nodeByFiberPosition(fibers.size());
    for (size_t i = 0; i < fibers.size(); ++i) {
        const auto tag = effectiveFiberHvTag(fibers[i].hv, fibers[i].manualHvTag);
        const std::string dir = fiberHvTagToPointCollectionString(tag);
        for (const auto& [position, ignored] : fiberPositions[i]) {
            (void)ignored;
            auto point = interpolateMappedPoint(
                i,
                exportData.atlas.fibers[i],
                position,
                dir,
                exportData.atlas.metadata,
                period);
            if (!point) {
                continue;
            }
            const size_t node = addGraphPoint(graph, idByKey, std::move(*point));
            nodeByFiberPosition[i].push_back(node);
        }
        std::sort(nodeByFiberPosition[i].begin(),
                  nodeByFiberPosition[i].end(),
                  [&](size_t a, size_t b) {
                      return graph.points[a].sourcePosition < graph.points[b].sourcePosition;
                  });
        for (size_t j = 1; j < nodeByFiberPosition[i].size(); ++j) {
            const size_t a = nodeByFiberPosition[i][j - 1];
            const size_t b = nodeByFiberPosition[i][j];
            const double credit =
                std::abs(graph.points[a].continuousWinding - graph.points[b].continuousWinding);
            addGraphEdge(graph, a, b, credit, true);
        }
    }

    auto nodeForEndpoint = [&](const AtlasLinkEndpoint& endpoint) -> std::optional<size_t> {
        for (size_t i = 0; i < exportData.atlas.fibers.size(); ++i) {
            if (atlasFiberPathKey(exportData.atlas.fibers[i].fiberPath) ==
                atlasFiberPathKey(endpoint.fiberPath)) {
                const auto snapped = nearestControlSourcePosition(
                    exportData.atlas.fibers[i],
                    static_cast<double>(endpoint.sourceIndex));
                if (!snapped) {
                    return std::nullopt;
                }
                const auto key = pointKey(i, *snapped);
                const auto it = idByKey.find(key);
                if (it != idByKey.end()) {
                    return it->second;
                }
            }
        }
        return std::nullopt;
    };
    for (const auto& link : exportData.atlas.links) {
        const auto a = nodeForEndpoint(link.first);
        const auto b = nodeForEndpoint(link.second);
        if (a && b) {
            addGraphEdge(graph, *a, *b, 0.0, false);
        }
    }
    for (const auto& link : temporaryLinks) {
        const auto a = idByKey.find(pointKey(link.firstFiber, link.firstPosition));
        const auto b = idByKey.find(pointKey(link.secondFiber, link.secondPosition));
        if (a != idByKey.end() && b != idByKey.end()) {
            addGraphEdge(graph, a->second, b->second, 0.0, false);
        }
    }
    report.temporaryLinks = temporaryLinks.size();
    return graph;
}

struct BeamState {
    std::vector<size_t> path;
    std::set<size_t> positiveEdges;
    double newCredit = 0.0;
    size_t zeroHops = 0;
};

std::string pathTieKey(const Graph& graph, const std::vector<size_t>& path)
{
    std::ostringstream out;
    for (size_t node : path) {
        out << graph.points[node].stableKey << ';';
    }
    return out.str();
}

bool betterState(const Graph& graph, const BeamState& a, const BeamState& b)
{
    if (std::abs(a.newCredit - b.newCredit) > kEpsilon) {
        return a.newCredit > b.newCredit;
    }
    if (a.zeroHops != b.zeroHops) {
        return a.zeroHops < b.zeroHops;
    }
    if (a.path.size() != b.path.size()) {
        return a.path.size() < b.path.size();
    }
    return pathTieKey(graph, a.path) < pathTieKey(graph, b.path);
}

std::optional<BeamState> bestLinePath(const Graph& graph,
                                      const std::vector<size_t>& starts,
                                      size_t beamWidth)
{
    std::vector<BeamState> beam;
    for (size_t start : starts) {
        beam.push_back({{start}, {}, 0.0, 0});
    }
    std::optional<BeamState> best;
    const size_t maxSteps = std::max<size_t>(1, graph.points.size());
    for (size_t step = 0; step < maxSteps && !beam.empty(); ++step) {
        std::vector<BeamState> next;
        for (const auto& state : beam) {
            if (state.newCredit > kEpsilon && (!best || betterState(graph, state, *best))) {
                best = state;
            }
            const size_t current = state.path.back();
            std::vector<size_t> edgeIds = graph.adjacency[current];
            std::sort(edgeIds.begin(), edgeIds.end(), [&](size_t a, size_t b) {
                const auto otherA = graph.edges[a].a == current ? graph.edges[a].b : graph.edges[a].a;
                const auto otherB = graph.edges[b].a == current ? graph.edges[b].b : graph.edges[b].a;
                return graph.points[otherA].stableKey < graph.points[otherB].stableKey;
            });
            for (size_t edgeId : edgeIds) {
                const auto& edge = graph.edges[edgeId];
                const size_t neighbor = edge.a == current ? edge.b : edge.a;
                if (std::find(state.path.begin(), state.path.end(), neighbor) != state.path.end()) {
                    continue;
                }
                BeamState advanced = state;
                advanced.path.push_back(neighbor);
                if (edge.positive && !edge.covered) {
                    if (advanced.positiveEdges.insert(edgeId).second) {
                        advanced.newCredit += edge.credit;
                    }
                } else {
                    ++advanced.zeroHops;
                }
                next.push_back(std::move(advanced));
            }
        }
        std::sort(next.begin(), next.end(), [&](const BeamState& a, const BeamState& b) {
            return betterState(graph, a, b);
        });
        if (next.size() > beamWidth) {
            next.resize(beamWidth);
        }
        beam = std::move(next);
    }
    return best;
}

std::vector<std::vector<size_t>> lineCoverPaths(Graph& graph, size_t beamWidth)
{
    std::vector<std::vector<size_t>> paths;
    while (true) {
        std::vector<size_t> starts;
        for (size_t i = 0; i < graph.edges.size(); ++i) {
            const auto& edge = graph.edges[i];
            if (edge.positive && !edge.covered) {
                starts.push_back(edge.a);
                starts.push_back(edge.b);
            }
        }
        if (starts.empty()) {
            break;
        }
        std::sort(starts.begin(), starts.end(), [&](size_t a, size_t b) {
            return graph.points[a].stableKey < graph.points[b].stableKey;
        });
        starts.erase(std::unique(starts.begin(), starts.end()), starts.end());
        auto best = bestLinePath(graph, starts, std::max<size_t>(1, beamWidth));
        if (!best || best->positiveEdges.empty()) {
            for (size_t i = 0; i < graph.edges.size(); ++i) {
                if (graph.edges[i].positive && !graph.edges[i].covered) {
                    best = BeamState{{graph.edges[i].a, graph.edges[i].b}, {i}, graph.edges[i].credit, 0};
                    break;
                }
            }
        }
        if (!best) {
            break;
        }
        for (size_t edgeId : best->positiveEdges) {
            graph.edges[edgeId].covered = true;
        }
        paths.push_back(std::move(best->path));
    }
    return paths;
}

std::vector<std::vector<size_t>> crossWindingChains(const Graph& graph,
                                                    const AtlasConstraintExportOptions& options)
{
    auto betterNext = [&](size_t candidate,
                          size_t currentBest,
                          double candidateError,
                          double bestError,
                          double candidateZDelta,
                          double bestZDelta) {
        if (std::abs(candidateError - bestError) > kEpsilon) {
            return candidateError < bestError;
        }
        if (std::abs(candidateZDelta - bestZDelta) > kEpsilon) {
            return candidateZDelta < bestZDelta;
        }
        return graph.points[candidate].stableKey < graph.points[currentBest].stableKey;
    };

    std::vector<std::vector<size_t>> candidates;
    for (size_t start = 0; start < graph.points.size(); ++start) {
        std::vector<size_t> chain;
        std::unordered_set<size_t> used;
        size_t current = start;
        chain.push_back(current);
        used.insert(current);

        while (true) {
            std::optional<size_t> best;
            double bestError = std::numeric_limits<double>::infinity();
            double bestZDelta = std::numeric_limits<double>::infinity();
            for (size_t next = 0; next < graph.points.size(); ++next) {
                if (used.count(next)) {
                    continue;
                }
                const double dw =
                    graph.points[next].continuousWinding - graph.points[current].continuousWinding;
                const double windingError = std::abs(dw - options.crossWindingTarget);
                const double zDelta =
                    std::abs(graph.points[next].world[2] - graph.points[current].world[2]);
                if (windingError > options.crossWindingTolerance ||
                    zDelta > options.crossZThreshold) {
                    continue;
                }
                if (!best ||
                    betterNext(next, *best, windingError, bestError, zDelta, bestZDelta)) {
                    best = next;
                    bestError = windingError;
                    bestZDelta = zDelta;
                }
            }
            if (!best) {
                break;
            }
            current = *best;
            chain.push_back(current);
            used.insert(current);
        }

        if (chain.size() >= 2) {
            candidates.push_back(std::move(chain));
        }
    }

    auto containedSet = [](const std::vector<size_t>& chain) {
        std::vector<size_t> nodes = chain;
        std::sort(nodes.begin(), nodes.end());
        nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
        return nodes;
    };
    std::sort(candidates.begin(), candidates.end(), [&](const auto& a, const auto& b) {
        if (a.size() != b.size()) {
            return a.size() > b.size();
        }
        const auto& firstA = graph.points[a.front()];
        const auto& firstB = graph.points[b.front()];
        if (std::abs(firstA.continuousWinding - firstB.continuousWinding) > kEpsilon) {
            return firstA.continuousWinding < firstB.continuousWinding;
        }
        return pathTieKey(graph, a) < pathTieKey(graph, b);
    });

    std::vector<std::vector<size_t>> chains;
    std::unordered_set<size_t> usedInKeptChain;
    for (auto& chain : candidates) {
        auto nodes = containedSet(chain);
        bool overlapsKept = false;
        for (size_t node : nodes) {
            if (usedInKeptChain.count(node)) {
                overlapsKept = true;
                break;
            }
        }
        if (overlapsKept) {
            continue;
        }
        for (size_t node : nodes) {
            usedInKeptChain.insert(node);
        }
        chains.push_back(std::move(chain));
    }

    std::sort(chains.begin(), chains.end(), [&](const auto& a, const auto& b) {
        return pathTieKey(graph, a) < pathTieKey(graph, b);
    });
    return chains;
}

uint64_t addCollectionForPath(PointCollections& collections,
                              const Graph& graph,
                              const std::string& name,
                              const std::vector<size_t>& path)
{
    if (path.empty()) {
        return 0;
    }
    const uint64_t collectionId = collections.addCollection(name);
    CollectionMetadata metadata;
    metadata.absolute_winding_number = false;
    collections.setCollectionMetadata(collectionId, metadata);
    std::vector<ColPoint> points;
    points.reserve(path.size());
    const double baseWinding = graph.points[path.front()].continuousWinding;
    for (size_t node : path) {
        const auto& gp = graph.points[node];
        ColPoint point = collections.addPoint(
            name,
            cv::Vec3f(static_cast<float>(gp.world[0]),
                      static_cast<float>(gp.world[1]),
                      static_cast<float>(gp.world[2])));
        point.creation_time = 0;
        point.winding_annotation = static_cast<float>(gp.continuousWinding - baseWinding);
        point.fiber_dir = gp.fiberDir;
        collections.updatePoint(point);
        points.push_back(point);
    }
    for (size_t i = 0; i < points.size(); ++i) {
        if (i > 0) {
            points[i].links.push_back(points[i - 1].id);
        }
        if (i + 1 < points.size()) {
            points[i].links.push_back(points[i + 1].id);
        }
        collections.updatePoint(points[i]);
    }
    return collectionId;
}

void linkGeneratedCollectionWindings(PointCollections& collections,
                                     const std::vector<uint64_t>& collectionIds)
{
    if (collectionIds.empty()) {
        return;
    }
    const uint64_t root = collectionIds.front();
    std::vector<uint64_t> rootLinks;
    rootLinks.reserve(collectionIds.size() - 1);
    for (size_t i = 1; i < collectionIds.size(); ++i) {
        rootLinks.push_back(collectionIds[i]);
    }
    collections.setCollectionWindingsLinked(root, rootLinks);
    for (size_t i = 1; i < collectionIds.size(); ++i) {
        collections.setCollectionWindingsLinked(collectionIds[i], {root});
    }
}

uint64_t mix64(uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

cv::Scalar debugColor(size_t groupIndex, uint64_t salt)
{
    const uint64_t value = mix64(static_cast<uint64_t>(groupIndex) ^ salt);
    const int b = 64 + static_cast<int>(value & 0x7f);
    const int g = 96 + static_cast<int>((value >> 8) & 0x9f);
    const int r = 96 + static_cast<int>((value >> 17) & 0x9f);
    return cv::Scalar(b, g, r);
}

struct DebugCanvasTransform {
    double minU = 0.0;
    double minV = 0.0;
    double scale = 1.0;
    int width = 0;
    int height = 0;
};

std::optional<DebugCanvasTransform> debugCanvasTransformForGraph(const Graph& graph)
{
    double minU = std::numeric_limits<double>::infinity();
    double minV = std::numeric_limits<double>::infinity();
    double maxU = -std::numeric_limits<double>::infinity();
    double maxV = -std::numeric_limits<double>::infinity();
    for (const auto& point : graph.points) {
        if (!std::isfinite(point.atlasU) ||
            !std::isfinite(point.atlasV)) {
            continue;
        }
        minU = std::min(minU, point.atlasU);
        minV = std::min(minV, point.atlasV);
        maxU = std::max(maxU, point.atlasU);
        maxV = std::max(maxV, point.atlasV);
    }
    if (!std::isfinite(minU) || !std::isfinite(minV) ||
        !std::isfinite(maxU) || !std::isfinite(maxV)) {
        return std::nullopt;
    }
    const double spanU = std::max(1.0, maxU - minU);
    const double spanV = std::max(1.0, maxV - minV);
    const double longSide = std::max(spanU, spanV);
    const double scale = std::clamp(1400.0 / longSide, 1.0, 32.0);
    DebugCanvasTransform transform;
    transform.minU = minU;
    transform.minV = minV;
    transform.scale = scale;
    transform.width = std::clamp(static_cast<int>(std::ceil(spanU * scale)) + 48, 256, 2048);
    transform.height = std::clamp(static_cast<int>(std::ceil(spanV * scale)) + 48, 256, 2048);
    return transform;
}

cv::Point debugPoint(const ExportPoint& point, const DebugCanvasTransform& transform)
{
    const int x = 24 + static_cast<int>(std::llround((point.atlasU - transform.minU) * transform.scale));
    const int y = transform.height - 24 -
        static_cast<int>(std::llround((point.atlasV - transform.minV) * transform.scale));
    return {std::clamp(x, 0, transform.width - 1),
            std::clamp(y, 0, transform.height - 1)};
}

cv::Point debugUvPoint(double u, double v, const DebugCanvasTransform& transform)
{
    const int x = 24 + static_cast<int>(std::llround((u - transform.minU) * transform.scale));
    const int y = transform.height - 24 -
        static_cast<int>(std::llround((v - transform.minV) * transform.scale));
    return {std::clamp(x, 0, transform.width - 1),
            std::clamp(y, 0, transform.height - 1)};
}

void drawDebugMarker(cv::Mat& image,
                     const cv::Point& point,
                     const cv::Scalar& color,
                     size_t groupIndex)
{
    constexpr int radius = 5;
    switch (groupIndex % 5) {
    case 0:
        cv::circle(image, point, radius, color, cv::FILLED, cv::LINE_AA);
        break;
    case 1:
        cv::rectangle(image,
                      {point.x - radius, point.y - radius},
                      {point.x + radius, point.y + radius},
                      color,
                      cv::FILLED,
                      cv::LINE_AA);
        break;
    case 2: {
        const cv::Point pts[] = {
            {point.x, point.y - radius - 1},
            {point.x - radius - 1, point.y + radius},
            {point.x + radius + 1, point.y + radius},
        };
        cv::fillConvexPoly(image, pts, 3, color, cv::LINE_AA);
        break;
    }
    case 3: {
        const cv::Point pts[] = {
            {point.x, point.y - radius - 1},
            {point.x - radius - 1, point.y},
            {point.x, point.y + radius + 1},
            {point.x + radius + 1, point.y},
        };
        cv::fillConvexPoly(image, pts, 4, color, cv::LINE_AA);
        break;
    }
    default:
        cv::line(image,
                 {point.x - radius, point.y - radius},
                 {point.x + radius, point.y + radius},
                 color,
                 2,
                 cv::LINE_AA);
        cv::line(image,
                 {point.x - radius, point.y + radius},
                 {point.x + radius, point.y - radius},
                 color,
                 2,
                 cv::LINE_AA);
        break;
    }
}

void drawDebugGroup(cv::Mat& image,
                    const Graph& graph,
                    const DebugCanvasTransform& transform,
                    const std::vector<size_t>& group,
                    size_t groupIndex,
                    uint64_t colorSalt,
                    bool connectSegments)
{
    const cv::Scalar color = debugColor(groupIndex, colorSalt);
    if (connectSegments) {
        for (size_t i = 1; i < group.size(); ++i) {
            const auto& a = graph.points[group[i - 1]];
            const auto& b = graph.points[group[i]];
            cv::line(image,
                     debugPoint(a, transform),
                     debugPoint(b, transform),
                     color,
                     2,
                     cv::LINE_AA);
        }
    }
    for (size_t node : group) {
        drawDebugMarker(image,
                        debugPoint(graph.points[node], transform),
                        color,
                        groupIndex);
    }
}

void drawDebugGroups(cv::Mat& image,
                     const Graph& graph,
                     const DebugCanvasTransform& transform,
                     const std::vector<std::vector<size_t>>& groups,
                     uint64_t colorSalt,
                     bool connectSegments)
{
    for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        drawDebugGroup(
            image, graph, transform, groups[groupIndex], groupIndex, colorSalt, connectSegments);
    }
}

void writeDebugMat(const cv::Mat& image, const fs::path& path)
{
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    std::vector<int> params;
    if (ext == ".tif" || ext == ".tiff") {
        params = {cv::IMWRITE_TIFF_COMPRESSION, cv::IMWRITE_TIFF_COMPRESSION_LZW};
    }
    if (!cv::imwrite(path.string(), image, params)) {
        throw std::runtime_error("failed to write debug image: " + path.string());
    }
}

double wrappedCrossDebugU(const Graph& graph, const ExportPoint& point)
{
    if (graph.periodColumns <= 0 || !std::isfinite(point.continuousWinding)) {
        return point.atlasU;
    }
    const double windingInPeriod =
        point.continuousWinding - std::floor(point.continuousWinding);
    return windingInPeriod * static_cast<double>(graph.periodColumns);
}

struct WrappedDebugPoint {
    double u = 0.0;
    double v = 0.0;
};

std::vector<WrappedDebugPoint> wrappedDebugGroupPoints(const Graph& graph,
                                                       const std::vector<size_t>& group,
                                                       bool minimizeLineLength)
{
    std::vector<WrappedDebugPoint> out;
    out.reserve(group.size());
    const double period = static_cast<double>(graph.periodColumns);
    for (size_t node : group) {
        if (node >= graph.points.size()) {
            continue;
        }
        const auto& point = graph.points[node];
        double u = wrappedCrossDebugU(graph, point);
        if (minimizeLineLength && period > 0.0 && !out.empty() && std::isfinite(u)) {
            const double shift = std::round((out.back().u - u) / period);
            u += shift * period;
        }
        out.push_back({u, point.atlasV});
    }
    return out;
}

std::optional<DebugCanvasTransform> debugWrappedTransform(
    const Graph& graph,
    const std::vector<std::vector<size_t>>& groups,
    bool minimizeGroupLineLength,
    const std::vector<std::vector<size_t>>& extraGroups = {},
    bool minimizeExtraGroupLineLength = false)
{
    double minU = std::numeric_limits<double>::infinity();
    double minV = std::numeric_limits<double>::infinity();
    double maxU = -std::numeric_limits<double>::infinity();
    double maxV = -std::numeric_limits<double>::infinity();
    auto accumulate = [&](const std::vector<std::vector<size_t>>& sourceGroups,
                          bool minimizeLineLength) {
        for (const auto& group : sourceGroups) {
            for (const auto& point : wrappedDebugGroupPoints(graph, group, minimizeLineLength)) {
                const double u = point.u;
                const double v = point.v;
                if (!std::isfinite(u) || !std::isfinite(v)) {
                    continue;
                }
                minU = std::min(minU, u);
                minV = std::min(minV, v);
                maxU = std::max(maxU, u);
                maxV = std::max(maxV, v);
            }
        }
    };
    accumulate(groups, minimizeGroupLineLength);
    accumulate(extraGroups, minimizeExtraGroupLineLength);
    if (!std::isfinite(minU) || !std::isfinite(minV) ||
        !std::isfinite(maxU) || !std::isfinite(maxV)) {
        return std::nullopt;
    }
    const double spanU = std::max(1.0, maxU - minU);
    const double spanV = std::max(1.0, maxV - minV);
    const double longSide = std::max(spanU, spanV);
    const double scale = std::clamp(1400.0 / longSide, 1.0, 64.0);
    DebugCanvasTransform transform;
    transform.minU = minU;
    transform.minV = minV;
    transform.scale = scale;
    transform.width = std::clamp(static_cast<int>(std::ceil(spanU * scale)) + 48, 128, 2048);
    transform.height = std::clamp(static_cast<int>(std::ceil(spanV * scale)) + 48, 128, 2048);
    return transform;
}

void drawWrappedDebugGroup(cv::Mat& image,
                           const Graph& graph,
                           const DebugCanvasTransform& transform,
                           const std::vector<size_t>& group,
                           size_t groupIndex,
                           uint64_t colorSalt,
                           bool connectSegments,
                           bool minimizeLineLength)
{
    const cv::Scalar color = debugColor(groupIndex, colorSalt);
    const auto points = wrappedDebugGroupPoints(graph, group, minimizeLineLength);
    if (connectSegments) {
        for (size_t i = 1; i < points.size(); ++i) {
            cv::line(image,
                     debugUvPoint(points[i - 1].u, points[i - 1].v, transform),
                     debugUvPoint(points[i].u, points[i].v, transform),
                     color,
                     2,
                     cv::LINE_AA);
        }
    }
    for (const auto& point : points) {
        drawDebugMarker(image,
                        debugUvPoint(point.u, point.v, transform),
                        color,
                        groupIndex);
    }
}

void drawWrappedDebugGroups(cv::Mat& image,
                            const Graph& graph,
                            const DebugCanvasTransform& transform,
                            const std::vector<std::vector<size_t>>& groups,
                            uint64_t colorSalt,
                            bool connectSegments,
                            bool minimizeLineLength)
{
    for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        drawWrappedDebugGroup(
            image,
            graph,
            transform,
            groups[groupIndex],
            groupIndex,
            colorSalt,
            connectSegments,
            minimizeLineLength);
    }
}

size_t writeDebugImage(const Graph& graph,
                       const std::vector<std::vector<size_t>>& lineGroups,
                       const std::vector<std::vector<size_t>>& crossGroups,
                       const fs::path& debugOutput)
{
    if (debugOutput.empty() || (lineGroups.empty() && crossGroups.empty())) {
        return 0;
    }
    const bool hasFormatExtension = debugOutput.has_extension();
    const fs::path debugDir = hasFormatExtension
        ? (debugOutput.parent_path().empty() ? fs::path(".") : debugOutput.parent_path())
        : debugOutput;
    const std::string extension = hasFormatExtension
        ? debugOutput.extension().string()
        : std::string(".png");

    std::error_code ec;
    fs::create_directories(debugDir, ec);
    if (ec) {
        throw std::runtime_error("failed to create debug image directory " +
                                 debugDir.string() + ": " + ec.message());
    }

    const auto transform = debugCanvasTransformForGraph(graph);
    if (!transform) {
        return 0;
    }
    cv::Mat image(transform->height, transform->width, CV_8UC3, cv::Scalar(0, 0, 0));
    drawDebugGroups(image, graph, *transform, lineGroups, 0x11efeULL, true);
    drawDebugGroups(image, graph, *transform, crossGroups, 0xc4055ULL, false);

    const fs::path path = hasFormatExtension
        ? debugOutput
        : (debugDir / ("atlas_constraints_debug" + extension));
    writeDebugMat(image, path);
    return 1;
}

cv::Mat blankDebugImage(const DebugCanvasTransform& transform)
{
    return cv::Mat(transform.height, transform.width, CV_8UC3, cv::Scalar(0, 0, 0));
}

size_t writeDebugDirectory(const Graph& graph,
                           const std::vector<std::vector<size_t>>& lineGroups,
                           const std::vector<std::vector<size_t>>& crossGroups,
                           const fs::path& debugDir)
{
    if (debugDir.empty() || (lineGroups.empty() && crossGroups.empty())) {
        return 0;
    }
    std::error_code ec;
    fs::create_directories(debugDir, ec);
    if (ec) {
        throw std::runtime_error("failed to create debug image directory " +
                                 debugDir.string() + ": " + ec.message());
    }

    const auto transform = debugCanvasTransformForGraph(graph);
    if (!transform) {
        return 0;
    }

    size_t written = 0;
    if (!lineGroups.empty()) {
        cv::Mat image = blankDebugImage(*transform);
        drawDebugGroups(image, graph, *transform, lineGroups, 0x11efeULL, true);
        writeDebugMat(image, debugDir / "lines.tif");
        ++written;
    }
    if (!crossGroups.empty()) {
        cv::Mat image = blankDebugImage(*transform);
        drawDebugGroups(image, graph, *transform, crossGroups, 0xc4055ULL, false);
        writeDebugMat(image, debugDir / "cross.tif");
        ++written;

        const auto wrappedTransform = debugWrappedTransform(graph, crossGroups, true);
        if (wrappedTransform) {
            cv::Mat wrappedImage = blankDebugImage(*wrappedTransform);
            drawWrappedDebugGroups(
                wrappedImage, graph, *wrappedTransform, crossGroups, 0x9c2055ULL, true, true);
            writeDebugMat(wrappedImage, debugDir / "cross_wrapped.tif");
            ++written;
        }
    }
    if (!lineGroups.empty() && !crossGroups.empty()) {
        const auto wrappedTransform = debugWrappedTransform(graph, lineGroups, false, crossGroups, true);
        if (wrappedTransform) {
            cv::Mat wrappedImage = blankDebugImage(*wrappedTransform);
            drawWrappedDebugGroups(
                wrappedImage, graph, *wrappedTransform, lineGroups, 0x11efeULL, true, false);
            drawWrappedDebugGroups(
                wrappedImage, graph, *wrappedTransform, crossGroups, 0x9c2055ULL, true, true);
            writeDebugMat(wrappedImage, debugDir / "lines_cross_wrapped.tif");
            ++written;
        }
    }

    for (size_t i = 0; i < lineGroups.size(); ++i) {
        cv::Mat image = blankDebugImage(*transform);
        drawDebugGroup(image, graph, *transform, lineGroups[i], i, 0x11efeULL, true);
        std::ostringstream name;
        name << "line_" << std::setw(6) << std::setfill('0') << i << ".tif";
        writeDebugMat(image, debugDir / name.str());
        ++written;
    }
    for (size_t i = 0; i < crossGroups.size(); ++i) {
        cv::Mat image = blankDebugImage(*transform);
        drawDebugGroup(image, graph, *transform, crossGroups[i], i, 0xc4055ULL, false);
        std::ostringstream name;
        name << "cross_" << std::setw(6) << std::setfill('0') << i << ".tif";
        writeDebugMat(image, debugDir / name.str());
        ++written;

        const std::vector<std::vector<size_t>> singleGroup = {crossGroups[i]};
        const auto wrappedTransform = debugWrappedTransform(graph, singleGroup, true);
        if (wrappedTransform) {
            cv::Mat wrappedImage = blankDebugImage(*wrappedTransform);
            drawWrappedDebugGroup(
                wrappedImage, graph, *wrappedTransform, crossGroups[i], i, 0x9c2055ULL, true, true);
            std::ostringstream wrappedName;
            wrappedName << "cross_wrapped_" << std::setw(6) << std::setfill('0') << i << ".tif";
            writeDebugMat(wrappedImage, debugDir / wrappedName.str());
            ++written;
        }
    }
    return written;
}

} // namespace

AtlasConstraintExportResult exportAtlasConstraints(
    const LasagnaAtlasExport& exportData,
    const QuadSurface* baseSurface,
    const vc::lasagna::LasagnaNormalSampler* windingSampler,
    const AtlasConstraintExportOptions& options)
{
    if (!baseSurface) {
        throw std::runtime_error("atlas constraints export requires a base surface");
    }
    AtlasConstraintExportResult result;
    result.report.atlasFibers = exportData.atlas.fibers.size();
    result.report.sourceLinks = exportData.atlas.links.size();

    std::vector<LoadedFiber> fibers;
    fibers.reserve(exportData.objects.size());
    std::unordered_map<std::string, size_t> fiberIndexByKey;
    for (size_t i = 0; i < exportData.objects.size(); ++i) {
        const auto& object = exportData.objects[i];
        fibers.push_back(loadFiberForConstraints(object.fiberPath, object.fiberRelativePath));
        fiberIndexByKey[atlasFiberPathKey(object.fiberRelativePath)] = i;
    }

    const auto temporaryLinks = temporaryCycleClosingLinks(
        exportData,
        *baseSurface,
        windingSampler,
        fibers,
        options,
        fiberIndexByKey);
    Graph graph = buildGraph(
        exportData,
        *baseSurface,
        fibers,
        temporaryLinks,
        options,
        result.report);

    std::vector<std::vector<size_t>> linePaths;
    std::vector<std::vector<size_t>> crossChains;
    std::vector<uint64_t> generatedCollectionIds;

    if (options.exportLineConstraints) {
        linePaths = lineCoverPaths(graph, options.greedyBeamWidth);
        size_t index = 0;
        for (const auto& path : linePaths) {
            std::ostringstream name;
            name << "atlas_line_" << std::setw(6) << std::setfill('0') << index++;
            const uint64_t collectionId =
                addCollectionForPath(result.collections, graph, name.str(), path);
            if (collectionId != 0) {
                generatedCollectionIds.push_back(collectionId);
            }
            ++result.report.lineCollections;
            result.report.linePoints += path.size();
        }
    }

    if (options.exportCrossWindingConstraints) {
        crossChains = crossWindingChains(graph, options);
        size_t index = 0;
        for (const auto& chain : crossChains) {
            std::ostringstream name;
            name << "atlas_cross_" << std::setw(6) << std::setfill('0') << index++;
            const uint64_t collectionId =
                addCollectionForPath(result.collections, graph, name.str(), chain);
            if (collectionId != 0) {
                generatedCollectionIds.push_back(collectionId);
            }
            ++result.report.crossCollections;
            result.report.crossPoints += chain.size();
        }
    }
    linkGeneratedCollectionWindings(result.collections, generatedCollectionIds);
    if (!options.debugImagesDir.empty()) {
        result.report.debugImagesWritten += writeDebugImage(
            graph,
            linePaths,
            crossChains,
            options.debugImagesDir);
    }
    if (!options.debugDirectory.empty()) {
        result.report.debugImagesWritten += writeDebugDirectory(
            graph,
            linePaths,
            crossChains,
            options.debugDirectory);
    }
    return result;
}

} // namespace vc::atlas
