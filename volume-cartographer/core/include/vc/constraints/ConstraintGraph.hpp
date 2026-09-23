#pragma once

#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <atomic>
#include <array>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>
#include "vc/constraints/SharedVector.hpp"

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
    int stripGeneration = 0; // zero denotes legacy straight fiber ribbons
    std::vector<cv::Vec3d> xyz;
    std::vector<cv::Vec2d> initialUv;
    std::vector<cv::Vec3i> triangles;
    // Empty for implicit grids; otherwise one byte per source cell.
    std::vector<unsigned char> quadMask;
    std::vector<double> lineArclength, columnArclength;
    std::vector<cv::Vec3d> line;
    std::vector<size_t> controls;
};
struct Placement {
    std::shared_ptr<const Node> source;
    SharedVector<cv::Vec2d> uv;
};
struct Graph {
    // Nonempty only for immutable theta/z display imports, never a fitted chart.
    nlohmann::json windingLayout;
    Settings settings;
    std::string seed;
    std::map<std::string, Placement> nodes;
    SharedVector<Link> links;
};
struct Candidate {
    std::string id;
    bool fiber = false;
    double distance = 0;
    std::vector<Link> contacts;
};
using DiscoveryBatch = std::function<void(std::vector<Candidate>)>;
struct RenderOwnership {
    cv::Mat_<int> nodes;
    std::vector<std::string> sourceIds;
};
struct ProjectionCollision {
    Location a, b;
    cv::Vec2d uv;
    cv::Vec3d xyzA, xyzB, normalA, normalB, tangentU, tangentV;
};
struct FitResult {
    Graph graph;
    std::shared_ptr<QuadSurface> surface;
    // Provenance of fixed-layout raster samples, retained for exact picking.
    std::shared_ptr<const RenderOwnership> ownership;
    std::string sessionRevision;
    cv::Vec2d origin{0, 0};
    double correspondenceRms = 0;
    double stretchP95 = 1;
    bool valid = false;
    // A bounded attachment needs an explicit full-graph refit. It is not a
    // verdict that the source geometry is invalid, and cannot be committed.
    bool needsRefit = false;
    // Bounds of the surface samples replaced by an incremental result.
    std::optional<std::array<cv::Vec2d,2>> changedSurfaceBounds;
    // Diagnostics for an incremental fit; zero for the global path.
    size_t localChartVertices = 0, localSurfaceSamples = 0, localExpansions = 0;
    // Source witnesses retained when a merged raster is blocked by unrelated
    // patches at the same UV address. Empty for ordinary workspace rendering.
    std::vector<ProjectionCollision> projectionCollisions;
    std::string message;
};

struct PreparedGraph;
// Contact indices are stable within this immutable committed revision.
struct GraphEdit {
    std::shared_ptr<const PreparedGraph> base;
    std::map<std::string,Placement> addedNodes;
    std::vector<Link> addedLinks;
    std::vector<size_t> removedLinks;
    // Transient fiber-review constraints in the accepted fiber's chart. Rebuilt
    // from accepted decisions, never embedded in a two-node association.
    struct FixedTarget { Location point; cv::Vec2d uv; };
    std::vector<FixedTarget> fixedTargets;
};
std::shared_ptr<const PreparedGraph> prepareGraph(const FitResult&, const std::atomic_bool* cancel = nullptr);
std::shared_ptr<const PreparedGraph> prepareGraph(const FitResult&, const GraphEdit&, const std::atomic_bool* cancel = nullptr);
const FitResult& preparedResult(const PreparedGraph&);
Graph materialize(const GraphEdit&);
// Adapter for callers editing an existing snapshot. The interactive proposal
// path constructs its delta directly, without diffing the committed graph.
GraphEdit graphEdit(std::shared_ptr<const PreparedGraph>, const Graph&);
void addKeypoint(GraphEdit&, Link);
void removeKeypoint(GraphEdit&, const Link&);
// Fiber review uses one proposal per patch with at least two dense contacts
// within radius, then fits all dense contacts within twice radius.
enum class ContactPolicy { LocalPlacement, FiberReview };
GraphEdit proposeEdit(const Catalog&, std::shared_ptr<const PreparedGraph>, const Candidate&, bool wholeComponent,
                      const lasagna::NormalSampler&, const std::atomic_bool* cancel = nullptr,
                      ContactPolicy = ContactPolicy::LocalPlacement);
enum class FitPolicy { Joint, FixedReference, FiberReview };
// FiberReview uses robust automatic contacts and validates an independent
// patch layer. Its surface contains that layer; graph retains the fixed fiber
// and proposal association. Fixed targets are transient review inputs only.
FitResult fitEdit(const GraphEdit&, const std::atomic_bool* cancel = nullptr, Progress = {},
                  FitPolicy = FitPolicy::Joint, double reviewRegularization = 1);
FitResult renderProposalSources(const GraphEdit&, const std::string& selected,
                                const std::atomic_bool* cancel = nullptr);

Catalog loadCatalog(const Settings&, const std::atomic_bool* cancel = nullptr, Progress = {});
// Audit original dense fibers and annotated links, independent of fitted UV.
// Centers and fibers must use the same coordinate frame. Nonzero winding
// closure needs review or a chart cut; it is not proof of an annotation error.
CycleAudit auditFiberCycles(const Catalog&, const std::vector<cv::Vec3d>& centers,
                           double minimumRadius, const std::atomic_bool* cancel = nullptr);
// Fiber strips are built by the solver-free marching ribbon below (makeFastFiberNode);
// this generation marks such strips, and saved strips of any other generation
// (legacy straight ribbons, the earlier robust Ceres fit) regenerate on load.
inline constexpr int kFiberStripGeneration = 100;
std::shared_ptr<const Node> makeFiberNode(const Fiber&, const Settings&, const lasagna::NormalSampler&, const std::atomic_bool* cancel = nullptr);
// Solver-free strip for automatic linking: the fiber resampled every
// settings.spacing along its arclength, with rows marched outward along the
// in-sheet cross direction while re-reading the normal field, then lightly
// smoothed along the fiber. It estimates where the fiber's sheet is across the
// ribbon without the robust fit's validity guarantees (folds are possible on
// bad normals), which the distance-based linker tolerates, and re-centres on
// the sampler's sheet score when it provides one. Seconds per thousand fibers
// rather than minutes per fiber. makeFiberNode is this builder with defaults.
struct FastStripOptions {
    double step = 5;              // marching step across the sheet, in graph voxels
    double followAngle = 35;      // degrees: a sampled normal within this of the current one is followed
    double normalSigma = 2;       // Gaussian sigma (columns) smoothing the per-column normals
    double offsetSigma = 1;       // Gaussian sigma (columns) smoothing each row's offsets
    double outlierAngle = 35;     // degrees: per-column normals farther than this from the smoothed field are interpolated
    // Optional sheet score (higher on the sheet, e.g. the Lasagna pred_dt
    // channel): every marched point moves along the local normal to the
    // nearest clear peak within snapRange, so the ribbon re-centres on the
    // sheet instead of drifting off it.
    std::function<std::optional<double>(const cv::Vec3d&)> sheetScore;
    double snapRange = 2, snapStep = 1;
    double snapMinContrast = 8;  // peak must exceed the value at the current point by this much
};
std::shared_ptr<const Node> makeFastFiberNode(const Fiber&, const Settings&, const lasagna::NormalSampler&, const std::atomic_bool* cancel = nullptr,
                                              const FastStripOptions& = {});
// Disk cache of fitted fiber strips. The key covers the fiber file's bytes, the
// strip settings and the robust strip version, so a hit reproduces
// makeFiberNode bit for bit while skipping its solver cascade (seconds to
// minutes per fiber). Absent or mismatched entries return null.
// method names the builder ("robust" for makeFiberNode, "fast" for makeFastFiberNode).
std::string fiberStripCacheKey(const Fiber&, const Settings&, const std::string& method = "robust");
std::shared_ptr<const Node> loadCachedFiberNode(const std::filesystem::path& directory, const Fiber&, const Settings&, const std::string& method = "robust");
void saveCachedFiberNode(const std::filesystem::path& directory, const Fiber&, const Settings&, const Node&, const std::string& method = "robust");
// Upgrade a saved graph transactionally; patch placements and fiber chart
// coordinates are preserved by interpolation on the old source triangles.
bool regenerateFiberStrips(Graph&, const lasagna::NormalSampler&, const std::atomic_bool* cancel = nullptr);
std::shared_ptr<const Node> loadPatchNode(const PatchEntry&, const Settings&);
Graph seedGraph(const Catalog&, size_t component, const Settings&, const lasagna::NormalSampler&,
                const std::atomic_bool* cancel = nullptr);
// Search every lineStride-th dense point, plus both endpoints. Use 1 for a full scan.
std::vector<Candidate> discover(const Catalog&, const Graph&, const std::string& selectedNode,
                                const std::atomic_bool* cancel = nullptr, Progress = {}, DiscoveryBatch = {}, int lineStride = 4,
                                const std::set<std::string>& excludedPatches = {},
                                ContactPolicy = ContactPolicy::LocalPlacement);
Graph propose(const Catalog&, const Graph&, const Candidate&, bool wholeComponent,
              const lasagna::NormalSampler&, const std::atomic_bool* cancel = nullptr);
FitResult fit(Graph, const std::atomic_bool* cancel = nullptr, Progress = {});
// Align source charts without creating a fused raster. Used by initialization
// and group-registration experiments; valid does not imply renderability.
// rigidOnly estimates one planar pose per source and preserves its chart shape.
FitResult fitSourceCharts(Graph, const std::atomic_bool* cancel = nullptr,
                          Progress = {}, bool rigidOnly = false);
// Fit an edit against a committed surface, preserving distant geometry. Expand
// the free region while its boundary remains strained; fall back to fit() for
// broad edits or changed global settings. Residuals and tolerances are shared.
// A rejection that persists across a doubled region is returned at once with
// localChartVertices > 0; callers wanting the full re-layout's verdict on it
// run fit() on the same draft.
FitResult fitIncremental(Graph, const FitResult& committed,
                         const std::atomic_bool* cancel = nullptr, Progress = {});
FitResult renderProposalSources(const Graph& draft, const Graph& committed, const std::string& selected,
                                const std::atomic_bool* cancel = nullptr);
// Temporary display of the initial attachment, without shared XYZ optimization.
// Never commit this result: fit() must still validate the joint solution.
FitResult renderPlacement(Graph, const std::atomic_bool* cancel = nullptr, Progress = {});
// Inspect immutable fiber/patch XYZ independently at the fitted chart positions.
// Display only: no shared-surface refinement and no change to the accepted fit.
FitResult renderSourceLayer(const Graph&, bool fibers, const std::atomic_bool* cancel = nullptr);
FitResult render(Graph, const std::atomic_bool* cancel = nullptr);
// Fuse registered charts with boundary-tapered weights and source-tangent
// seam fitting. Fiber strips participate only when explicitly in the graph.
// Keeps shared-grid geometry validation and does not fill unsupported gaps.
// Opt-in: ordinary workspace rendering retains its existing weighting.
FitResult renderMergedSurface(Graph, const std::atomic_bool* cancel = nullptr);
// Strict-quad, edge-connected components; single-quad islands are omitted.
// Crop each result to its own bounds without creating or interpolating samples.
std::vector<FitResult> splitSurfaceComponents(const FitResult&, size_t minimumQuads = 2);
cv::Vec2d placedLocation(const Graph&, const Location&);
cv::Vec3d sourceLocation(const Node&, const cv::Vec2d&);
std::optional<Location> pick(const Graph&, const cv::Vec2d& graphUv,
                             const std::string& onlyNode = {});
Location fiberLineLocation(const Node&, size_t lineIndex);

// Session embeds immutable loaded geometry so a preview is never reconstructed
// against silently changed input files. Catalog paths remain for further growth.
nlohmann::json settingsToJson(const Settings&);
Settings settingsFromJson(const nlohmann::json&);
nlohmann::json locationToJson(const Location&);
Location locationFromJson(const nlohmann::json&);
nlohmann::json graphToJson(const Graph&);
Graph graphFromJson(const nlohmann::json&);
void saveSession(const FitResult&, const std::filesystem::path&);
FitResult loadSession(const std::filesystem::path&);
std::optional<Location> pickRendered(const FitResult&, const cv::Vec2d& graphUv);
std::filesystem::path quarantineDirectory(const std::filesystem::path& workspace);
// Moves complete source folders, excludes all their components, and publishes
// the saved winding workspace only after the replacement surface is valid.
FitResult quarantinePatches(const FitResult&, const std::filesystem::path& workspace,
                            const std::vector<std::string>& sourceIds, const Progress& = {});
void exportSurface(const FitResult&, const std::filesystem::path&);
} // namespace vc::constraints
