#pragma once

#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <atomic>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

class QuadSurface;
namespace vc::lasagna { class NormalSampler; }

namespace vc::constraints {
using Progress = std::function<void(const std::string&)>;
struct Settings {
    double spacing = 20;
    double radius = 4;
    double fiberScale = 0.25;
    double patchScale = 1;
    double normalScale = 4;
    double volumeScale = 1;
    double stiffness = 1;
    std::string normalManifest;
    std::filesystem::path fiberFolder, patchFolder;
    void validate() const;
};

// Stable coordinates in the original source grid, independent of fitted UV
// placement and output bounds. Fibers use their derived, persisted ribbon grid.
struct Location {
    std::string node;
    cv::Vec2d grid{0, 0};
    int64_t lineIndex = -1;
};
struct Link {
    Location a, b;
    bool keypoint = false;
    bool pending = false;
};
struct FiberLink {
    std::string a, b;
    size_t controlA = 0, controlB = 0;
    bool pending = false;
};
struct Fiber {
    std::string id;
    std::filesystem::path path;
    char direction = '?';
    std::vector<cv::Vec3d> line;
    std::vector<size_t> controls;
    // False for interpolation/legacy geometry whose winding is uncertain.
    bool fullyTraced = false;
};
struct CycleIssue {
    long long turns = 0;
    bool reliable = false;
    // Indices into Catalog::links; the entire fundamental cycle, not blame
    // assigned to whichever edge happened to close the spanning tree.
    std::vector<size_t> links;
};
struct CycleAudit {
    bool available = false;
    size_t cycles = 0, evaluatedCycles = 0, checkedCycles = 0, uncertainCycles = 0;
    size_t skippedLinks = 0;
    std::vector<CycleIssue> issues;
    std::string summary;
    std::vector<std::string> details;
};
struct PatchEntry {
    std::string id;
    std::filesystem::path path;
    cv::Vec3d low, high;
};
struct Catalog {
    std::map<std::string, Fiber> fibers;
    std::vector<FiberLink> links;
    std::vector<std::vector<std::string>> components;
    std::vector<PatchEntry> patches;
    std::vector<std::string> warnings;
    CycleAudit cycleAudit;
    struct BoundsIndex;
    std::shared_ptr<const BoundsIndex> boundsIndex;
};

struct Node {
    std::string id;
    std::filesystem::path path;
    bool fiber = false;
    char direction = '?';
    int rows = 0, cols = 0;
    std::vector<cv::Vec3d> xyz;
    std::vector<cv::Vec2d> initialUv;
    std::vector<cv::Vec3i> triangles;
    std::vector<double> lineArclength, columnArclength;
    std::vector<cv::Vec3d> line;
    std::vector<size_t> controls;
};
struct Placement {
    std::shared_ptr<const Node> source;
    std::vector<cv::Vec2d> uv;
};
struct Graph {
    Settings settings;
    std::string seed;
    std::map<std::string, Placement> nodes;
    std::vector<Link> links;
};
struct Candidate {
    std::string id;
    bool fiber = false;
    double distance = 0;
    std::vector<Link> contacts;
};
using DiscoveryBatch = std::function<void(std::vector<Candidate>)>;
struct FitResult {
    Graph graph;
    std::shared_ptr<QuadSurface> surface;
    cv::Vec2d origin{0, 0};
    double correspondenceRms = 0;
    double stretchP95 = 1;
    bool valid = false;
    // Diagnostics for an incremental fit; zero for the global path.
    size_t localChartVertices = 0, localSurfaceSamples = 0, localExpansions = 0;
    std::string message;
};

Catalog loadCatalog(const Settings&, const std::atomic_bool* cancel = nullptr, Progress = {});
// Audit original dense fibers and annotated links, independent of fitted UV.
// Centers and fibers must use the same coordinate frame. Nonzero winding
// closure needs review or a chart cut; it is not proof of an annotation error.
CycleAudit auditFiberCycles(const Catalog&, const std::vector<cv::Vec3d>& centers,
                           double minimumRadius, const std::atomic_bool* cancel = nullptr);
std::shared_ptr<const Node> makeFiberNode(const Fiber&, const Settings&, const lasagna::NormalSampler&);
std::shared_ptr<const Node> loadPatchNode(const PatchEntry&, const Settings&);
Graph seedGraph(const Catalog&, size_t component, const Settings&, const lasagna::NormalSampler&,
                const std::atomic_bool* cancel = nullptr);
// Search every lineStride-th dense point, plus both endpoints. Use 1 for a full scan.
std::vector<Candidate> discover(const Catalog&, const Graph&, const std::string& selectedNode,
                                const std::atomic_bool* cancel = nullptr, Progress = {}, DiscoveryBatch = {}, int lineStride = 4);
Graph propose(const Catalog&, const Graph&, const Candidate&, bool wholeComponent,
              const lasagna::NormalSampler&, const std::atomic_bool* cancel = nullptr);
FitResult fit(Graph, const std::atomic_bool* cancel = nullptr, Progress = {});
// Fit an edit against a committed surface, preserving distant geometry. Expand
// the free region when its boundary remains strained; fall back to fit() for
// broad edits or changed global settings. Residuals and tolerances are shared.
FitResult fitIncremental(Graph, const FitResult& committed,
                         const std::atomic_bool* cancel = nullptr, Progress = {});
FitResult renderProposalSources(const Graph& draft, const Graph& committed, const std::string& selected,
                                const std::atomic_bool* cancel = nullptr);
// Temporary display of the initial attachment, without shared XYZ optimization.
// Never commit this result: fit() must still validate the joint solution.
FitResult renderPlacement(Graph, const std::atomic_bool* cancel = nullptr);
// Inspect immutable fiber/patch XYZ independently at the fitted chart positions.
// Display only: no shared-surface refinement and no change to the accepted fit.
FitResult renderSourceLayer(const Graph&, bool fibers, const std::atomic_bool* cancel = nullptr);
FitResult render(Graph, const std::atomic_bool* cancel = nullptr);
cv::Vec2d placedLocation(const Graph&, const Location&);
cv::Vec3d sourceLocation(const Node&, const cv::Vec2d&);
std::optional<Location> pick(const Graph&, const cv::Vec2d& graphUv,
                             const std::string& onlyNode = {});
Location fiberLineLocation(const Node&, size_t lineIndex);

// Session embeds immutable loaded geometry so a preview is never reconstructed
// against silently changed input files. Catalog paths remain for further growth.
nlohmann::json graphToJson(const Graph&);
Graph graphFromJson(const nlohmann::json&);
void saveSession(const FitResult&, const std::filesystem::path&);
FitResult loadSession(const std::filesystem::path&);
void exportSurface(const FitResult&, const std::filesystem::path&);
} // namespace vc::constraints
