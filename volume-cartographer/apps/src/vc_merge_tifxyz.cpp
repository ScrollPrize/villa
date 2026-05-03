// vc_merge_tifxyz
//
// N-surface global tifxyz merge. Only required input is
// --merge <path/to/volpkg/merge.json>: surfaces and edges come from the
// row-major grid in that file (cells are full tifxyz directory names under
// <volpkg>/paths/, null/"" = empty). Output dirs are auto-named under
// <volpkg>/paths/: <alpha_first>_merged for the final tifxyz and
// <alpha_first>_merged_prestraightened for the intermediate, where
// <alpha_first> is the alphabetically smallest surface name in the grid.
// Tunables: --ref, --strength-u/-v, --ransac-* are flags; the remaining
// stage params (ridge_lambda, consensus_*, idw_k, step_size, anchor_bin_size)
// are hard-coded. For each edge: SurfacePatchIndex::locate-based overlap,
// real-overlap mask, stratified anchors, RANSAC similarity. Across all
// edges: joint per-surface affine bundle adjustment, per-surface TPS RBF,
// N-way EDT blend, OBJ emit, and rasterization via vc_obj2tifxyz_legacy.
// Final stage: gentle UV re-parameterization (umbilicus theta unwrap +
// global affine) and a second rasterization pass for a clean iso-theta /
// iso-z grid.

#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/flann.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <boost/program_options.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace po = boost::program_options;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

// Anchor-selection internals: the per-edge threshold sweep, density-filtered
// match mask, normal-agreement gate, and minimum connected-component size are
// fixed heuristics — never tuned per-config in practice. Hardcoded here.
constexpr float kThresholds[]    = {4.0f, 5.0f, 6.0f, 7.0f};
constexpr float kNormalThresh    = 0.85f;
constexpr int   kDensityRadius   = 7;
constexpr float kDensityThresh   = 0.5f;
constexpr int   kMinComponent    = 200;

struct OverlapMaps {
    cv::Mat_<float>   distance;   // -1 where no match, else euclidean distance
    cv::Mat_<float>   normAgree;  // NaN where no match, else |nA . nB|
    cv::Mat_<uint8_t> mask;       // 0/1 raw match mask
};

OverlapMaps computeOverlap(const std::shared_ptr<QuadSurface>& a,
                           const std::shared_ptr<QuadSurface>& b,
                           const SurfacePatchIndex& idx,
                           float threshold)
{
    const cv::Mat_<cv::Vec3f> pts = a->rawPoints();
    const int H = pts.rows;
    const int W = pts.cols;

    OverlapMaps m;
    m.distance  = cv::Mat_<float>(H, W, -1.0f);
    m.normAgree = cv::Mat_<float>(H, W, std::numeric_limits<float>::quiet_NaN());
    m.mask      = cv::Mat_<uint8_t>(H, W, (uint8_t)0);

    // Warm the grid-normal cache on A and ensure B's points are loaded so the
    // parallel loop below is a pure read on both surfaces.
    (void)a->gridNormal(0, 0);
    (void)b->rawPoints();

    #pragma omp parallel for schedule(dynamic, 16)
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            const cv::Vec3f& p = pts(r, c);
            if (p[0] < 0) continue;

            auto res = idx.locate(p, threshold, b);
            if (!res || res->distance >= threshold) continue;

            m.mask(r, c) = 1;
            m.distance(r, c) = res->distance;

            const cv::Vec3f nA = a->gridNormal(r, c);
            const cv::Vec3f nB = b->normal(res->ptr);
            if (!std::isnan(nA[0]) && !std::isnan(nB[0])) {
                m.normAgree(r, c) = std::abs(nA.dot(nB));
            }
        }
    }
    return m;
}

cv::Mat_<float> boxDensity(const cv::Mat_<uint8_t>& mask, int radius)
{
    cv::Mat_<float> src;
    mask.convertTo(src, CV_32F);
    cv::Mat_<float> dst;
    const int k = 2 * radius + 1;
    cv::boxFilter(src, dst, CV_32F, cv::Size(k, k), cv::Point(-1, -1),
                  /*normalize=*/true, cv::BORDER_REFLECT);
    return dst;
}

int largestComponent(cv::Mat_<uint8_t>& mask, int minSize)
{
    cv::Mat labels, stats, cent;
    const int n = cv::connectedComponentsWithStats(mask, labels, stats, cent, 8, CV_32S);
    int kept = 0;
    // Label 0 is background.
    std::vector<uint8_t> keep(n, 0);
    for (int i = 1; i < n; ++i) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= minSize) {
            keep[i] = 1;
            ++kept;
        }
    }
    for (int y = 0; y < mask.rows; ++y) {
        const int* lrow = labels.ptr<int>(y);
        uint8_t* mrow = mask.ptr<uint8_t>(y);
        for (int x = 0; x < mask.cols; ++x) {
            if (mrow[x] && !keep[lrow[x]]) mrow[x] = 0;
        }
    }
    return kept;
}

struct Stats {
    long   matches = 0;
    long   valid = 0;
    double distMedian = std::numeric_limits<double>::quiet_NaN();
    double distMean   = std::numeric_limits<double>::quiet_NaN();
    double normMedian = std::numeric_limits<double>::quiet_NaN();
    double normMean   = std::numeric_limits<double>::quiet_NaN();
    long   realCells  = 0;   // after normal + density + connected-component filter
    long   realComponents = 0;
};

// Derive a match mask at threshold t from a distance map produced at a larger
// threshold. locate() returns the true nearest-neighbor distance (capped at
// its own tolerance), so any cell with recorded distance in [0, t) is a match
// at threshold t. Cells with no match at the larger threshold stay zero.
cv::Mat_<uint8_t> maskAtThreshold(const cv::Mat_<float>& dist, float t)
{
    cv::Mat_<uint8_t> m(dist.size(), (uint8_t)0);
    for (int y = 0; y < dist.rows; ++y) {
        for (int x = 0; x < dist.cols; ++x) {
            const float d = dist(y, x);
            if (d >= 0.0f && d < t) m(y, x) = 1;
        }
    }
    return m;
}

Stats summarize(const cv::Mat_<cv::Vec3f>& pts,
                const cv::Mat_<float>& distMap,
                const cv::Mat_<float>& normMap,
                const cv::Mat_<uint8_t>& matchMask,
                cv::Mat_<uint8_t>& realMask)
{
    Stats s;
    std::vector<double> dists;
    std::vector<double> norms;
    dists.reserve(1 << 14);
    norms.reserve(1 << 14);
    for (int y = 0; y < pts.rows; ++y) {
        for (int x = 0; x < pts.cols; ++x) {
            if (pts(y, x)[0] < 0) continue;
            ++s.valid;
            if (matchMask(y, x)) {
                ++s.matches;
                dists.push_back(distMap(y, x));
                if (!std::isnan(normMap(y, x))) norms.push_back(normMap(y, x));
            }
        }
    }
    auto med = [](std::vector<double>& v) {
        if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    auto mean = [](const std::vector<double>& v) {
        if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
        double s = 0; for (double x : v) s += x; return s / v.size();
    };
    s.distMean   = mean(dists);
    s.distMedian = med(dists);
    s.normMean   = mean(norms);
    s.normMedian = med(norms);

    cv::Mat_<float> density = boxDensity(matchMask, kDensityRadius);
    realMask = cv::Mat_<uint8_t>(matchMask.size(), (uint8_t)0);
    for (int y = 0; y < matchMask.rows; ++y) {
        for (int x = 0; x < matchMask.cols; ++x) {
            if (!matchMask(y, x)) continue;
            if (density(y, x) < kDensityThresh) continue;
            const float na = normMap(y, x);
            if (std::isnan(na) || na < kNormalThresh) continue;
            realMask(y, x) = 255;
        }
    }
    s.realComponents = largestComponent(realMask, kMinComponent);
    s.realCells = cv::countNonZero(realMask);
    return s;
}

// -----------------------------------------------------------------------------
// Phase 2 (step 1): anchor placement.
// Stratified spatial binning over A's best-threshold real-overlap mask:
// divide the overlap bounding box into `binSize`-sided cells and, within each
// cell, keep the A-grid vertex with the smallest locate() distance to B. This
// biases selection toward the most trustworthy correspondences while forcing
// coverage across the whole strip. For each kept vertex we re-query the patch
// index to recover B's sub-pixel grid coord and world point -- those pairs are
// the inputs to the later rigid + non-rigid alignment stages.
// -----------------------------------------------------------------------------

struct Anchor {
    float     a_row = 0, a_col = 0;     // grid cell in A (integer when A
                                        // seeds, sub-pixel when B seeds)
    cv::Vec3f a_world{0, 0, 0};
    float     b_row = 0, b_col = 0;     // grid cell in B (sub-pixel when A
                                        // seeds, integer when B seeds)
    cv::Vec3f b_world{0, 0, 0};
    float     distance = 0;
    float     normal_agree = std::numeric_limits<float>::quiet_NaN();
};

std::vector<Anchor> pickAnchors(const std::shared_ptr<QuadSurface>& A,
                                const std::shared_ptr<QuadSurface>& B,
                                const SurfacePatchIndex& idx,
                                const cv::Mat_<uint8_t>& realMaskA,
                                const cv::Mat_<float>&   distMapA,
                                const cv::Mat_<float>&   normMapA,
                                int binSize,
                                float locateTolerance)
{
    const cv::Mat_<cv::Vec3f> ptsA = A->rawPoints();
    const int H = realMaskA.rows;
    const int W = realMaskA.cols;

    int ymin = H, ymax = -1, xmin = W, xmax = -1;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (!realMaskA(y, x)) continue;
            if (y < ymin) ymin = y;
            if (y > ymax) ymax = y;
            if (x < xmin) xmin = x;
            if (x > xmax) xmax = x;
        }
    }
    if (ymax < 0 || binSize <= 0) return {};

    const int nBy = std::max(1, (ymax - ymin + binSize) / binSize);
    const int nBx = std::max(1, (xmax - xmin + binSize) / binSize);

    std::vector<Anchor> anchors;
    anchors.reserve(static_cast<size_t>(nBy) * nBx);

    for (int by = 0; by < nBy; ++by) {
        const int y0 = ymin + by * binSize;
        const int y1 = std::min(H, y0 + binSize);
        for (int bx = 0; bx < nBx; ++bx) {
            const int x0 = xmin + bx * binSize;
            const int x1 = std::min(W, x0 + binSize);

            int   bestR = -1, bestC = -1;
            float bestD = std::numeric_limits<float>::infinity();
            for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) {
                    if (!realMaskA(y, x)) continue;
                    const float d = distMapA(y, x);
                    if (d < 0) continue;
                    if (d < bestD) { bestD = d; bestR = y; bestC = x; }
                }
            }
            if (bestR < 0) continue;

            const cv::Vec3f& aw = ptsA(bestR, bestC);
            auto res = idx.locate(aw, locateTolerance, B);
            if (!res || res->distance >= locateTolerance) continue;

            const cv::Vec2f bg = B->ptrToGrid(res->ptr);
            Anchor a;
            a.a_row   = bestR;
            a.a_col   = bestC;
            a.a_world = aw;
            a.b_col   = bg[0];
            a.b_row   = bg[1];
            a.b_world = B->coord(res->ptr);
            a.distance     = res->distance;
            a.normal_agree = normMapA(bestR, bestC);
            anchors.push_back(a);
        }
    }
    return anchors;
}

// =============================================================================
// Global merge: N-surface joint alignment + EDT blend, driven by a JSON config
// of surfaces+edges. Per-edge anchors and real-overlap masks are computed in
// memory using the helpers above (computeOverlap / summarize / pickAnchors);
// only the per-run diagnostics JSON and the final merged tifxyz hit disk.
// =============================================================================

using Mat1f = cv::Mat_<float>;
using Mat1b = cv::Mat_<uint8_t>;

struct GMSurfaceSpec { std::string name; fs::path path; };
struct GMEdgeSpec    { std::string a; std::string b; };

// Hard-coded stage params (no flag, no JSON). Match the long-tuned values that
// have been working across volpkgs; expose only the knobs that actually vary.
constexpr double   kRidgeLambda        = 100.0;
constexpr double   kConsensusC         = 6.0;
constexpr int      kConsensusMinActive = 3;
constexpr int      kIdwK               = 4;
constexpr int      kStepSize           = 20;
constexpr int      kAnchorBinSize      = 1;

struct GMConfig {
    fs::path paths_dir;
    std::string ref;            // empty => auto (largest by valid cells)
    // Per-edge RANSAC similarity (CLI-overridable).
    int      ransac_iters{3000};
    double   ransac_min_thresh{5.0};
    double   ransac_max_thresh{10.0};
    double   ransac_mad_k{3.0};
    uint32_t ransac_seed{0};
    // Straighten stage (gentle UV re-parameterization).
    double   straighten_strength_u{0.3};
    double   straighten_strength_v{0.5};
    std::vector<GMSurfaceSpec> surfaces;
    std::vector<GMEdgeSpec>    edges;
};

// -----------------------------------------------------------------------------
// Grid input: <volpkg>/merge.json holds a `rows` array. Each row is either a
// JSON array of strings or (preferred, easier to hand-edit) a single
// whitespace-delimited string. Each cell is a full tifxyz directory name
// under <paths_dir>/, or null/"" for an empty slot. Surfaces are deduplicated;
// edges are derived from horizontal + vertical adjacency, skipping self-edges
// and edges incident to empty cells. The duplicate-cell case (same name twice)
// resolves to the same surface and produces no self-edge, so adjacent
// neighbors above and beside still connect.
// -----------------------------------------------------------------------------

void gmResolveGrid(const fs::path& merge_path,
                   const fs::path& paths_dir,
                   std::vector<GMSurfaceSpec>& surfaces,
                   std::vector<GMEdgeSpec>& edges)
{
    const fs::path& mj = merge_path;
    std::ifstream f(mj);
    if (!f) throw std::runtime_error("cannot open " + mj.string());
    json j; f >> j;

    if (j.size() != 1 || !j.contains("rows"))
        throw std::runtime_error(mj.string() +
            ": only the 'rows' key is accepted");
    const auto& rows_j = j.at("rows");
    if (!rows_j.is_array() || rows_j.empty())
        throw std::runtime_error(mj.string() + ": 'rows' must be a non-empty array");

    const size_t R = rows_j.size();
    std::vector<std::vector<std::string>> grid(R);
    std::unordered_map<std::string, fs::path> name_to_path;

    // Each row may be either a JSON array of strings (legacy) OR a single
    // whitespace-delimited string. The string form is easier to hand-edit:
    //   "rows": ["name_a name_b name_c", "name_d name_e name_f"]
    auto splitWhitespace = [](const std::string& s) {
        std::vector<std::string> out;
        std::istringstream iss(s);
        std::string tok;
        while (iss >> tok) out.push_back(std::move(tok));
        return out;
    };

    for (size_t r = 0; r < R; ++r) {
        const auto& row_j = rows_j[r];
        std::vector<std::string> row_names;
        if (row_j.is_array()) {
            row_names.reserve(row_j.size());
            for (size_t c = 0; c < row_j.size(); ++c) {
                const auto& cell = row_j[c];
                if (cell.is_null()) { row_names.emplace_back(); continue; }
                if (!cell.is_string())
                    throw std::runtime_error(mj.string() + ": rows[" +
                        std::to_string(r) + "][" + std::to_string(c) +
                        "] must be a string or null");
                row_names.push_back(cell.get<std::string>());
            }
        } else if (row_j.is_string()) {
            row_names = splitWhitespace(row_j.get<std::string>());
        } else {
            throw std::runtime_error(mj.string() + ": rows[" + std::to_string(r) +
                "] must be an array or a whitespace-delimited string");
        }

        const size_t C = row_names.size();
        grid[r].resize(C);
        for (size_t c = 0; c < C; ++c) {
            const std::string& name = row_names[c];
            if (name.empty()) continue;
            if (name_to_path.count(name)) {
                grid[r][c] = name;  // duplicate cell -> same surface
                continue;
            }
            const fs::path dir = paths_dir / name;
            if (!fs::is_directory(dir))
                throw std::runtime_error(mj.string() + ": rows[" +
                    std::to_string(r) + "][" + std::to_string(c) + "] = '" +
                    name + "' is not a directory under " + paths_dir.string());
            name_to_path[name] = dir;
            grid[r][c] = name;
        }
    }

    surfaces.clear();
    surfaces.reserve(name_to_path.size());
    std::set<std::string> seen;
    for (size_t r = 0; r < R; ++r) {
        for (size_t c = 0; c < grid[r].size(); ++c) {
            const std::string& n = grid[r][c];
            if (n.empty() || !seen.insert(n).second) continue;
            GMSurfaceSpec s;
            s.name = n;
            s.path = name_to_path.at(n);
            surfaces.push_back(std::move(s));
        }
    }

    if (surfaces.size() < 2)
        throw std::runtime_error(mj.string() + ": need at least 2 distinct "
            "surfaces in the grid; got " + std::to_string(surfaces.size()));

    auto edgeKey = [](const std::string& a, const std::string& b) {
        return a < b ? a + "\t" + b : b + "\t" + a;
    };
    std::set<std::string> edge_seen;
    edges.clear();
    auto addEdge = [&](const std::string& a, const std::string& b) {
        if (a.empty() || b.empty() || a == b) return;
        if (edge_seen.insert(edgeKey(a, b)).second) edges.push_back({a, b});
    };
    for (size_t r = 0; r < R; ++r) {
        const auto& row = grid[r];
        for (size_t c = 0; c + 1 < row.size(); ++c) addEdge(row[c], row[c + 1]);
    }
    for (size_t r = 0; r + 1 < R; ++r) {
        const size_t C = std::min(grid[r].size(), grid[r + 1].size());
        for (size_t c = 0; c < C; ++c) addEdge(grid[r][c], grid[r + 1][c]);
    }
}

void gmCheckConnected(const std::vector<GMSurfaceSpec>& surfaces,
                      const std::vector<GMEdgeSpec>& edges)
{
    std::unordered_map<std::string, std::vector<std::string>> adj;
    for (const auto& s : surfaces) adj[s.name];
    for (const auto& e : edges) {
        adj[e.a].push_back(e.b);
        adj[e.b].push_back(e.a);
    }
    std::unordered_set<std::string> visited;
    std::deque<std::string> q;
    q.push_back(surfaces.front().name);
    visited.insert(surfaces.front().name);
    while (!q.empty()) {
        const std::string u = q.front(); q.pop_front();
        for (const auto& v : adj[u])
            if (visited.insert(v).second) q.push_back(v);
    }
    if (visited.size() == surfaces.size()) return;
    std::ostringstream msg;
    msg << "edge graph from merge.json is disconnected; "
           "unreachable surfaces:";
    for (const auto& s : surfaces)
        if (!visited.count(s.name)) msg << "\n  " << s.name;
    throw std::runtime_error(msg.str());
}

// -----------------------------------------------------------------------------
// Surface holder: rawPoints from QuadSurface (already mask.tif-aware) split
// into x/y/z + a 0/1 mask derived from the -1 sentinel. Keeping the
// QuadSurface alive gives us SurfacePatchIndex access for pairwise overlap.
// -----------------------------------------------------------------------------

struct GMSurface {
    std::string name;
    fs::path path;
    std::shared_ptr<QuadSurface> qs;
    Mat1f x, y, z;
    Mat1b mask;
    int   H{0}, W{0};
    int   valid{0};
};

GMSurface gmLoadSurface(const GMSurfaceSpec& spec)
{
    auto up = load_quad_from_tifxyz(spec.path.string());
    if (!up) throw std::runtime_error("failed to load surface: " + spec.path.string());
    GMSurface g;
    g.name = spec.name;
    g.path = spec.path;
    g.qs = std::shared_ptr<QuadSurface>(up.release());
    const cv::Mat_<cv::Vec3f> pts = g.qs->rawPoints();
    g.H = pts.rows;
    g.W = pts.cols;
    g.x.create(g.H, g.W);
    g.y.create(g.H, g.W);
    g.z.create(g.H, g.W);
    g.mask = Mat1b::zeros(g.H, g.W);
    int valid = 0;
    for (int r = 0; r < g.H; ++r) {
        const cv::Vec3f* p = pts.ptr<cv::Vec3f>(r);
        float* xp = g.x.ptr<float>(r);
        float* yp = g.y.ptr<float>(r);
        float* zp = g.z.ptr<float>(r);
        uint8_t* mp = g.mask.ptr<uint8_t>(r);
        for (int c = 0; c < g.W; ++c) {
            xp[c] = p[c][0];
            yp[c] = p[c][1];
            zp[c] = p[c][2];
            const bool ok = !(xp[c] == -1.0f && yp[c] == -1.0f && zp[c] == -1.0f);
            mp[c] = ok ? 1 : 0;
            if (ok) ++valid;
        }
    }
    g.valid = valid;
    return g;
}

// -----------------------------------------------------------------------------
// Per-edge overlap + anchor selection. Runs the threshold sweep for one (A,B)
// edge and returns in-memory anchor pairs + per-surface real-overlap masks.
// No files written.
// -----------------------------------------------------------------------------

struct EdgeAnchors {
    std::vector<Anchor> anchors;
    Mat1b               realA, realB;     // 0/255
    float               threshold{0};
    double              score{0};
    int                 best_idx{-1};
    json                edge_json;        // diagnostics for summary
};

EdgeAnchors gmComputeEdgeAnchors(const std::shared_ptr<QuadSurface>& A,
                                 const std::shared_ptr<QuadSurface>& B,
                                 const std::string& a_name,
                                 const std::string& b_name,
                                 const SurfacePatchIndex& idx,
                                 const GMConfig& cfg)
{
    EdgeAnchors R;
    constexpr int kNumThresholds = sizeof(kThresholds) / sizeof(kThresholds[0]);
    const float maxThreshold = kThresholds[kNumThresholds - 1];

    OverlapMaps mAB = computeOverlap(A, B, idx, maxThreshold);
    OverlapMaps mBA = computeOverlap(B, A, idx, maxThreshold);

    auto toJson = [](const Stats& s) {
        return json{
            {"valid_cells", s.valid},
            {"match_cells", s.matches},
            {"distance_mean",   s.distMean},
            {"distance_median", s.distMedian},
            {"normal_agree_mean",   s.normMean},
            {"real_overlap_cells", s.realCells},
            {"real_overlap_components", s.realComponents},
        };
    };

    json perThreshold = json::array();
    int    bestIdx = -1;
    double bestScore = -1.0;
    long   bestRealCells = -1;
    Mat1b  bestRealA, bestRealB;
    // One-sided fallback for thin/asymmetric overlaps where one surface is
    // dense enough to pass the density+component filter but the other is too
    // sparse. We seed anchor extraction from whichever side survives.
    int    oneIdx = -1;
    long   oneRealCells = -1;
    Mat1b  oneRealA, oneRealB;
    bool   oneSideA = true;
    for (int i = 0; i < kNumThresholds; ++i) {
        const float t = kThresholds[i];
        Mat1b maskA = maskAtThreshold(mAB.distance, t);
        Mat1b maskB = maskAtThreshold(mBA.distance, t);
        Mat1b realA, realB;
        Stats sA = summarize(A->rawPoints(), mAB.distance, mAB.normAgree, maskA, realA);
        Stats sB = summarize(B->rawPoints(), mBA.distance, mBA.normAgree, maskB, realB);
        const double pA = sA.matches > 0 ? double(sA.realCells)/sA.matches : 0.0;
        const double pB = sB.matches > 0 ? double(sB.realCells)/sB.matches : 0.0;
        const double score = std::sqrt(pA * pB);
        const long minReal = std::min(sA.realCells, sB.realCells);
        const long maxReal = std::max(sA.realCells, sB.realCells);
        const bool aOk = sA.realComponents >= 1 && sA.realCells >= kMinComponent;
        const bool bOk = sB.realComponents >= 1 && sB.realCells >= kMinComponent;
        const bool valid = aOk && bOk;
        perThreshold.push_back({
            {"threshold", t}, {"purity_A", pA}, {"purity_B", pB},
            {"score", score}, {"valid", valid},
            {"A", toJson(sA)}, {"B", toJson(sB)},
        });
        if (valid && (score > bestScore
                      || (score == bestScore && minReal > bestRealCells))) {
            bestScore = score;
            bestRealCells = minReal;
            bestIdx = i;
            bestRealA = realA.clone();
            bestRealB = realB.clone();
        }
        if (!valid && (aOk || bOk) && maxReal > oneRealCells) {
            oneIdx = i;
            oneRealCells = maxReal;
            oneRealA = realA.clone();
            oneRealB = realB.clone();
            oneSideA = sA.realCells >= sB.realCells;
        }
    }
    bool useSideA = true;
    if (bestIdx < 0 && oneIdx >= 0) {
        bestIdx = oneIdx;
        bestRealCells = oneRealCells;
        bestRealA = std::move(oneRealA);
        bestRealB = std::move(oneRealB);
        useSideA = oneSideA;
        std::cout << "    one-sided fallback: anchors from "
                  << (useSideA ? a_name : b_name) << " side ("
                  << bestRealCells << " real cells)\n";
    }
    if (bestIdx < 0) {
        std::ostringstream msg;
        msg << "no valid threshold for edge "
            << a_name << "<->" << b_name << ":";
        for (const auto& pt : perThreshold) {
            const auto& Aj = pt.at("A");
            const auto& Bj = pt.at("B");
            msg << "\n      t=" << pt.at("threshold").get<double>()
                << ": A.match=" << Aj.at("match_cells").get<long>()
                << " A.real="   << Aj.at("real_overlap_cells").get<long>()
                << " A.comp="   << Aj.at("real_overlap_components").get<long>()
                << " | B.match=" << Bj.at("match_cells").get<long>()
                << " B.real="   << Bj.at("real_overlap_cells").get<long>()
                << " B.comp="   << Bj.at("real_overlap_components").get<long>();
        }
        throw std::runtime_error(msg.str());
    }

    const float bestT = kThresholds[bestIdx];
    std::vector<Anchor> anchors;
    if (useSideA) {
        anchors = pickAnchors(A, B, idx, bestRealA,
                              mAB.distance, mAB.normAgree,
                              kAnchorBinSize, bestT);
    } else {
        anchors = pickAnchors(B, A, idx, bestRealB,
                              mBA.distance, mBA.normAgree,
                              kAnchorBinSize, bestT);
        // pickAnchors fills a_* with the seed-side coords. We seeded from B,
        // so swap so the Anchor convention (a_* = A, b_* = B) holds.
        for (auto& a : anchors) {
            std::swap(a.a_row, a.b_row);
            std::swap(a.a_col, a.b_col);
            std::swap(a.a_world, a.b_world);
        }
    }

    R.anchors   = std::move(anchors);
    R.realA     = std::move(bestRealA);
    R.realB     = std::move(bestRealB);
    R.threshold = bestT;
    R.score     = bestScore;
    R.best_idx  = bestIdx;
    R.edge_json = {
        {"a", a_name}, {"b", b_name},
        {"best_threshold", bestT},
        {"best_score", bestScore},
        {"per_threshold", perThreshold},
        {"anchor_count", (int)R.anchors.size()},
        {"anchor_bin_size", kAnchorBinSize},
        {"anchor_seed_side", useSideA ? "A" : "B"},
    };
    return R;
}

// -----------------------------------------------------------------------------
// Global merge stages 2-6 (ported from vc_global_merge.cpp):
//   2. per-edge RANSAC similarity (outlier filter)
//   3. joint similarity fit (complex a*z+b LS, optional ridge, ref pinned)
//   4. per-surface TPS RBF on union of incident midpoint residuals
//   4.5 remap UVs into reference frame
//   5. N-way EDT blend with Tukey-IRLS consensus
//   6. emit OBJ + shell out to vc_obj2tifxyz_legacy
// -----------------------------------------------------------------------------

inline cv::Vec2d gmApplyAffine(const cv::Matx23d& M, const cv::Vec2d& p)
{
    return cv::Vec2d(M(0,0)*p[0] + M(0,1)*p[1] + M(0,2),
                     M(1,0)*p[0] + M(1,1)*p[1] + M(1,2));
}

cv::Matx23d gmFitSimilarity(const std::vector<cv::Vec2d>& src,
                            const std::vector<cv::Vec2d>& dst)
{
    const int N = (int)src.size();
    cv::Matx23d M = cv::Matx23d::eye();
    if (N == 0) return M;
    cv::Vec2d mu_s(0,0), mu_d(0,0);
    for (int i = 0; i < N; ++i) { mu_s += src[i]; mu_d += dst[i]; }
    mu_s *= 1.0 / N; mu_d *= 1.0 / N;
    Eigen::Matrix2d C = Eigen::Matrix2d::Zero();
    double var_s = 0.0;
    for (int i = 0; i < N; ++i) {
        Eigen::Vector2d X(src[i][0] - mu_s[0], src[i][1] - mu_s[1]);
        Eigen::Vector2d Y(dst[i][0] - mu_d[0], dst[i][1] - mu_d[1]);
        C += Y * X.transpose();
        var_s += X.squaredNorm();
    }
    C /= (double)N;
    var_s /= (double)N;
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(C, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2d U = svd.matrixU();
    Eigen::Matrix2d V = svd.matrixV();
    Eigen::Vector2d S = svd.singularValues();
    Eigen::Matrix2d D = Eigen::Matrix2d::Identity();
    if (U.determinant() * V.determinant() < 0) D(1,1) = -1.0;
    Eigen::Matrix2d R = U * D * V.transpose();
    double scale = (var_s > 0.0) ? (S(0)*D(0,0) + S(1)*D(1,1)) / var_s : 1.0;
    Eigen::Matrix2d sR = scale * R;
    Eigen::Vector2d mu_s_v(mu_s[0], mu_s[1]);
    Eigen::Vector2d mu_d_v(mu_d[0], mu_d[1]);
    Eigen::Vector2d t = mu_d_v - sR * mu_s_v;
    M(0,0) = sR(0,0); M(0,1) = sR(0,1); M(0,2) = t(0);
    M(1,0) = sR(1,0); M(1,1) = sR(1,1); M(1,2) = t(1);
    return M;
}

inline double gmSimilarityScale(const cv::Matx23d& M)
{
    return std::sqrt(std::abs(M(0,0)*M(1,1) - M(0,1)*M(1,0)));
}

inline std::pair<double,double> gmAffineScales(const cv::Matx23d& M)
{
    return {std::hypot(M(0,0), M(1,0)), std::hypot(M(0,1), M(1,1))};
}

double gmMedianInPlace(std::vector<double>& v)
{
    if (v.empty()) return 0.0;
    std::nth_element(v.begin(), v.begin() + v.size()/2, v.end());
    return v[v.size()/2];
}

struct GMRansac {
    cv::Matx23d M;
    std::vector<uint8_t> inlier;
    double thresh{0.0};
    double sigma_in{1.0};
    int n_in{0};
};

GMRansac gmRansacSimilarity(const std::vector<cv::Vec2d>& src,
                            const std::vector<cv::Vec2d>& dst,
                            int iters, double min_t, double max_t,
                            double mad_k, uint32_t seed)
{
    const int N = (int)src.size();
    GMRansac R; R.inlier.assign(N, 1);
    if (N < 2) {
        R.M = cv::Matx23d::eye();
        R.thresh = min_t; R.n_in = N; return R;
    }
    cv::Matx23d M0 = gmFitSimilarity(src, dst);
    std::vector<double> r0(N);
    for (int i = 0; i < N; ++i) r0[i] = cv::norm(dst[i] - gmApplyAffine(M0, src[i]));
    auto r0sorted = r0;
    double med = gmMedianInPlace(r0sorted);
    std::vector<double> dev(N);
    for (int i = 0; i < N; ++i) dev[i] = std::abs(r0[i] - med);
    double mad = gmMedianInPlace(dev) * 1.4826;
    double thresh = std::min(max_t, std::max(min_t, mad_k * mad));

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> uni(0, N - 1);
    int best_cnt = -1;
    std::vector<uint8_t> best_inl(N, 0);
    std::vector<cv::Vec2d> s2(2), d2(2);
    for (int it = 0; it < iters; ++it) {
        int i = uni(rng);
        int j = uni(rng);
        while (j == i) j = uni(rng);
        s2[0] = src[i]; s2[1] = src[j];
        d2[0] = dst[i]; d2[1] = dst[j];
        cv::Matx23d Mh;
        try { Mh = gmFitSimilarity(s2, d2); } catch (...) { continue; }
        int cnt = 0;
        std::vector<uint8_t> inl(N, 0);
        for (int k = 0; k < N; ++k) {
            const double r = cv::norm(dst[k] - gmApplyAffine(Mh, src[k]));
            if (r < thresh) { inl[k] = 1; ++cnt; }
        }
        if (cnt > best_cnt) { best_cnt = cnt; best_inl.swap(inl); }
    }
    auto gather = [&](std::vector<cv::Vec2d>& sv, std::vector<cv::Vec2d>& dv){
        sv.clear(); dv.clear();
        for (int k = 0; k < N; ++k) if (best_inl[k]) { sv.push_back(src[k]); dv.push_back(dst[k]); }
    };
    std::vector<cv::Vec2d> s_inl, d_inl;
    gather(s_inl, d_inl);
    cv::Matx23d M = gmFitSimilarity(s_inl, d_inl);
    for (int k = 0; k < N; ++k) {
        const double r = cv::norm(dst[k] - gmApplyAffine(M, src[k]));
        best_inl[k] = (r < thresh) ? 1 : 0;
    }
    gather(s_inl, d_inl);
    M = gmFitSimilarity(s_inl, d_inl);
    double sse = 0.0; int Nin = 0;
    for (int k = 0; k < N; ++k) {
        if (!best_inl[k]) continue;
        const double r = cv::norm(dst[k] - gmApplyAffine(M, src[k]));
        sse += r*r; ++Nin;
    }
    R.M = M;
    R.inlier = std::move(best_inl);
    R.thresh = thresh;
    R.sigma_in = (Nin > 0) ? std::sqrt(sse / Nin) : 1.0;
    R.n_in = Nin;
    return R;
}

struct GMEdgeRun {
    std::string a, b;
    std::vector<cv::Vec2d> pA, pB;
    cv::Matx23d M_pair;
    double sigma_in{1.0};
    double thresh{0.0};
    int n_in{0}, n_total{0};
    int real_overlap_a{-1}, real_overlap_b{-1};
};

// Joint per-surface AFFINE bundle adjustment: 6 dof per surface (m00, m01, t0,
// m10, m11, t1), ref pinned to identity. Each anchor pair on edge (a, b)
// contributes 2 scalar equations (one per output component). The x-system
// involves only (m00, m01, t0) and the y-system only (m10, m11, t1), so they
// decouple into two independent 3N-unknown LS problems. Edge equations are
// scaled by 1/max(σ_in, 1) so noisy edges don't drag clean ones off their
// fits. Tikhonov ridge pulls the linear part toward identity (m00→1, m01→0,
// m10→0, m11→1); translation rows are unregularized.
std::unordered_map<std::string, cv::Matx23d>
gmJointAffine(const std::vector<std::string>& names,
              const std::vector<GMEdgeRun>& edges,
              const std::string& ref,
              double ridge_lambda)
{
    std::vector<std::string> non_ref;
    non_ref.reserve(names.size());
    for (const auto& n : names) if (n != ref) non_ref.push_back(n);
    std::unordered_map<std::string,int> idx;
    for (size_t i = 0; i < non_ref.size(); ++i) idx[non_ref[i]] = (int)i;
    const int N = (int)non_ref.size();
    const int n_unk = 3 * N;  // (m_a0, m_a1, t) per non-ref surface, per axis system

    int n_data = 0;
    for (const auto& e : edges) n_data += (int)e.pA.size();
    const int n_ridge = (ridge_lambda > 0.0 && N > 0) ? 2 * N : 0;
    const int n_rows = n_data + n_ridge;

    std::vector<Eigen::Triplet<double>> Tx, Ty;
    Tx.reserve((size_t)n_data * 6 + n_ridge);
    Ty.reserve((size_t)n_data * 6 + n_ridge);
    Eigen::VectorXd bx(n_rows), by(n_rows);
    bx.setZero(); by.setZero();

    int row = 0;
    for (const auto& e : edges) {
        const double w = 1.0 / std::max(e.sigma_in, 1.0);
        const bool aRef = (e.a == ref);
        const bool bRef = (e.b == ref);
        const int ia = aRef ? -1 : idx[e.a];
        const int ib = bRef ? -1 : idx[e.b];
        for (size_t i = 0; i < e.pA.size(); ++i) {
            const double pAx = e.pA[i][0], pAy = e.pA[i][1];
            const double pBx = e.pB[i][0], pBy = e.pB[i][1];
            // X eq: (m00·pAx + m01·pAy + t0)_a − (m00·pBx + m01·pBy + t0)_b = 0
            // Y eq: same with (m10, m11, t1) on each side.
            double bx_rhs = 0.0, by_rhs = 0.0;
            if (aRef) {
                bx_rhs += pAx;
                by_rhs += pAy;
            } else {
                Tx.emplace_back(row, 3*ia + 0,  w * pAx);
                Tx.emplace_back(row, 3*ia + 1,  w * pAy);
                Tx.emplace_back(row, 3*ia + 2,  w);
                Ty.emplace_back(row, 3*ia + 0,  w * pAx);
                Ty.emplace_back(row, 3*ia + 1,  w * pAy);
                Ty.emplace_back(row, 3*ia + 2,  w);
            }
            if (bRef) {
                bx_rhs -= pBx;
                by_rhs -= pBy;
            } else {
                Tx.emplace_back(row, 3*ib + 0, -w * pBx);
                Tx.emplace_back(row, 3*ib + 1, -w * pBy);
                Tx.emplace_back(row, 3*ib + 2, -w);
                Ty.emplace_back(row, 3*ib + 0, -w * pBx);
                Ty.emplace_back(row, 3*ib + 1, -w * pBy);
                Ty.emplace_back(row, 3*ib + 2, -w);
            }
            bx(row) = -w * bx_rhs;
            by(row) = -w * by_rhs;
            ++row;
        }
    }
    if (n_ridge > 0) {
        for (int i = 0; i < N; ++i) {
            // x-system: penalize (m00 − 1) and m01.   y-system: m10 and (m11 − 1).
            Tx.emplace_back(row + 0, 3*i + 0, ridge_lambda);
            Tx.emplace_back(row + 1, 3*i + 1, ridge_lambda);
            bx(row + 0) = ridge_lambda;       // toward m00 = 1
            bx(row + 1) = 0.0;                // toward m01 = 0
            Ty.emplace_back(row + 0, 3*i + 0, ridge_lambda);
            Ty.emplace_back(row + 1, 3*i + 1, ridge_lambda);
            by(row + 0) = 0.0;                // toward m10 = 0
            by(row + 1) = ridge_lambda;       // toward m11 = 1
            row += 2;
        }
    }

    auto solve = [&](const std::vector<Eigen::Triplet<double>>& T,
                     const Eigen::VectorXd& rhs) -> Eigen::VectorXd {
        Eigen::SparseMatrix<double> A(n_rows, n_unk);
        A.setFromTriplets(T.begin(), T.end());
        A.makeCompressed();
        Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
        solver.compute(A);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("gm: affine factor failed");
        return solver.solve(rhs);
    };
    Eigen::VectorXd sx_ = solve(Tx, bx);
    Eigen::VectorXd sy_ = solve(Ty, by);

    std::unordered_map<std::string, cv::Matx23d> M_out;
    M_out[ref] = cv::Matx23d::eye();
    for (const auto& n : non_ref) {
        int i = idx[n];
        cv::Matx23d M;
        M(0,0) = sx_(3*i + 0); M(0,1) = sx_(3*i + 1); M(0,2) = sx_(3*i + 2);
        M(1,0) = sy_(3*i + 0); M(1,1) = sy_(3*i + 1); M(1,2) = sy_(3*i + 2);
        M_out[n] = M;
    }
    return M_out;
}

inline double gmTpsKernel(double r2)
{
    if (r2 <= 0.0) return 0.0;
    return r2 * 0.5 * std::log(r2);
}

struct GMSimRBF {
    cv::Matx23d M{cv::Matx23d::eye()};
    Eigen::MatrixXd anchors;        // N x 2
    Eigen::MatrixXd weights;        // N x 2
    Eigen::MatrixXd poly;           // 3 x 2
    int n_anchors{0};
    double resid_rms{0.0};
    double resid_max{0.0};

    void evalGrid(int H, int W, Mat1f& outX, Mat1f& outY) const
    {
        outX.create(H, W);
        outY.create(H, W);
        for (int r = 0; r < H; ++r) {
            float* px = outX.ptr<float>(r);
            float* py = outY.ptr<float>(r);
            for (int c = 0; c < W; ++c) {
                px[c] = (float)(M(0,0)*c + M(0,1)*r + M(0,2));
                py[c] = (float)(M(1,0)*c + M(1,1)*r + M(1,2));
            }
        }
        if (n_anchors == 0) return;
        const int N = n_anchors;
        std::vector<double> ax(N), ay(N), wx(N), wy(N);
        for (int i = 0; i < N; ++i) {
            ax[i] = anchors(i,0); ay[i] = anchors(i,1);
            wx[i] = weights(i,0); wy[i] = weights(i,1);
        }
        const double a0x = poly(0,0), axx = poly(1,0), ayx = poly(2,0);
        const double a0y = poly(0,1), axy = poly(1,1), ayy = poly(2,1);
        #pragma omp parallel for schedule(static) if(H*W > 4096)
        for (int r = 0; r < H; ++r) {
            float* px = outX.ptr<float>(r);
            float* py = outY.ptr<float>(r);
            for (int c = 0; c < W; ++c) {
                double dx = a0x + axx*c + ayx*r;
                double dy = a0y + axy*c + ayy*r;
                for (int i = 0; i < N; ++i) {
                    const double ddc = c - ax[i];
                    const double ddr = r - ay[i];
                    const double k = gmTpsKernel(ddc*ddc + ddr*ddr);
                    dx += wx[i] * k;
                    dy += wy[i] * k;
                }
                px[c] += (float)dx;
                py[c] += (float)dy;
            }
        }
    }
};

GMSimRBF gmBuildSurfaceWarp(const cv::Matx23d& M,
                            const std::vector<cv::Vec2d>& src_in,
                            const std::vector<cv::Vec2d>& resid_in)
{
    GMSimRBF S; S.M = M;
    if (src_in.empty()) { S.poly = Eigen::MatrixXd::Zero(3,2); return S; }

    std::unordered_map<int64_t, std::pair<cv::Vec2d, cv::Vec2d>> sums;
    std::unordered_map<int64_t, int> counts;
    sums.reserve(src_in.size() * 2);
    counts.reserve(src_in.size() * 2);
    for (size_t i = 0; i < src_in.size(); ++i) {
        const cv::Vec2d& s = src_in[i];
        const int64_t qc = (int64_t)std::llround(s[0] * 2.0);
        const int64_t qr = (int64_t)std::llround(s[1] * 2.0);
        const int64_t key = (qc << 32) ^ (qr & 0xffffffff);
        auto& acc = sums[key];
        acc.first  += s;
        acc.second += resid_in[i];
        counts[key]++;
    }
    const int N = (int)sums.size();
    Eigen::MatrixXd src(N, 2), res(N, 2);
    int row = 0;
    for (auto& kv : sums) {
        const int cnt = counts[kv.first];
        const cv::Vec2d ss = kv.second.first  * (1.0/cnt);
        const cv::Vec2d rr = kv.second.second * (1.0/cnt);
        src(row, 0) = ss[0]; src(row, 1) = ss[1];
        res(row, 0) = rr[0]; res(row, 1) = rr[1];
        ++row;
    }
    const int Na = N + 3;
    Eigen::MatrixXd Aug = Eigen::MatrixXd::Zero(Na, Na);
    Eigen::MatrixXd RHS = Eigen::MatrixXd::Zero(Na, 2);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const double dc = src(i,0) - src(j,0);
            const double dr = src(i,1) - src(j,1);
            Aug(i,j) = gmTpsKernel(dc*dc + dr*dr);
        }
        Aug(i, N+0) = 1.0; Aug(i, N+1) = src(i,0); Aug(i, N+2) = src(i,1);
        Aug(N+0, i) = 1.0; Aug(N+1, i) = src(i,0); Aug(N+2, i) = src(i,1);
        RHS(i, 0) = res(i, 0); RHS(i, 1) = res(i, 1);
    }
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(Aug);
    Eigen::MatrixXd sol = lu.solve(RHS);
    S.anchors = src;
    S.weights = sol.topRows(N);
    S.poly    = sol.bottomRows(3);
    S.n_anchors = N;
    double sse = 0.0, mx = 0.0;
    for (int i = 0; i < N; ++i) {
        const double rr = std::hypot(res(i,0), res(i,1));
        sse += rr*rr;
        if (rr > mx) mx = rr;
    }
    S.resid_rms = std::sqrt(sse / std::max(1, N));
    S.resid_max = mx;
    return S;
}

std::unordered_map<std::string, GMSimRBF>
gmBuildSurfaceWarps(const std::vector<std::string>& names,
                    const std::vector<GMEdgeRun>& edges,
                    const std::unordered_map<std::string, cv::Matx23d>& M_dict)
{
    std::unordered_map<std::string, std::vector<cv::Vec2d>> anchors_per, resid_per;
    for (const auto& n : names) {
        anchors_per[n].reserve(64);
        resid_per[n].reserve(64);
    }
    for (const auto& e : edges) {
        const auto& Ma = M_dict.at(e.a);
        const auto& Mb = M_dict.at(e.b);
        for (size_t i = 0; i < e.pA.size(); ++i) {
            const cv::Vec2d Ta_pA = gmApplyAffine(Ma, e.pA[i]);
            const cv::Vec2d Tb_pB = gmApplyAffine(Mb, e.pB[i]);
            const cv::Vec2d mid = 0.5 * (Ta_pA + Tb_pB);
            anchors_per[e.a].push_back(e.pA[i]);
            resid_per [e.a].push_back(mid - Ta_pA);
            anchors_per[e.b].push_back(e.pB[i]);
            resid_per [e.b].push_back(mid - Tb_pB);
        }
    }
    std::unordered_map<std::string, GMSimRBF> warps;
    for (const auto& n : names)
        warps[n] = gmBuildSurfaceWarp(M_dict.at(n), anchors_per[n], resid_per[n]);
    return warps;
}

struct GMUV { Mat1f uc, ur; };
struct GMBlend { Mat1f X, Y, Z; };

void gmRasterize(int GH, int GW, int u_min, int v_min,
                 const Mat1b& mvalid, const Mat1b& raw_native,
                 const Mat1f& uc, const Mat1f& ur, Mat1b& dst)
{
    dst = Mat1b::zeros(GH, GW);
    const int H = uc.rows, W = uc.cols;
    for (int r = 0; r < H; ++r) {
        const uint8_t* mv = mvalid.ptr<uint8_t>(r);
        const uint8_t* mr = raw_native.empty() ? nullptr : raw_native.ptr<uint8_t>(r);
        const float* uu = uc.ptr<float>(r);
        const float* vv = ur.ptr<float>(r);
        for (int c = 0; c < W; ++c) {
            if (!mv[c]) continue;
            if (mr && !mr[c]) continue;
            if (!std::isfinite(uu[c]) || !std::isfinite(vv[c])) continue;
            const int gu = (int)std::lround(uu[c]) - u_min;
            const int gv = (int)std::lround(vv[c]) - v_min;
            if (gu < 0 || gu >= GW || gv < 0 || gv >= GH) continue;
            dst(gv, gu) = 1;
        }
    }
    cv::Mat krn = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3,3));
    cv::dilate(dst, dst, krn, cv::Point(-1,-1), 1);
}

cv::Mat gmDistTransform(const Mat1b& bin)
{
    cv::Mat src;
    bin.convertTo(src, CV_8U, 255.0);
    cv::Mat dist;
    cv::distanceTransform(src, dist, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    cv::Mat dist64;
    dist.convertTo(dist64, CV_64F);
    return dist64;
}

Mat1f gmSampleBilinearReplicate(const cv::Mat& grid64,
                                const Mat1f& uc, const Mat1f& ur,
                                int u_min, int v_min)
{
    cv::Mat g32; grid64.convertTo(g32, CV_32F);
    Mat1f mapX(uc.size()), mapY(uc.size());
    for (int r = 0; r < uc.rows; ++r) {
        const float* uu = uc.ptr<float>(r);
        const float* vv = ur.ptr<float>(r);
        float* mx = mapX.ptr<float>(r);
        float* my = mapY.ptr<float>(r);
        for (int c = 0; c < uc.cols; ++c) {
            mx[c] = uu[c] - (float)u_min;
            my[c] = vv[c] - (float)v_min;
        }
    }
    Mat1f out(uc.size());
    cv::remap(g32, out, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return out;
}

std::unordered_map<std::string, GMBlend>
gmEdtBlend(const std::vector<std::string>& names,
           const std::unordered_map<std::string, GMSurface>& surfs,
           const std::unordered_map<std::string, GMUV>& uv,
           int u_min, int u_max, int v_min, int v_max,
           const std::unordered_map<std::string, Mat1b>& real_overlap_native,
           double consensus_c, int consensus_min_active, int idw_k)
{
    const int GW = u_max - u_min + 1;
    const int GH = v_max - v_min + 1;

    std::unordered_map<std::string, cv::Mat> DS_active, DS_valid;
    for (const auto& n : names) {
        const GMSurface& s = surfs.at(n);
        const Mat1f& uc = uv.at(n).uc;
        const Mat1f& ur = uv.at(n).ur;
        Mat1b mvalid(s.H, s.W);
        for (int r = 0; r < s.H; ++r) {
            const uint8_t* mp = s.mask.ptr<uint8_t>(r);
            const float* uu = uc.ptr<float>(r);
            const float* vv = ur.ptr<float>(r);
            uint8_t* dp = mvalid.ptr<uint8_t>(r);
            for (int c = 0; c < s.W; ++c)
                dp[c] = (mp[c] && std::isfinite(uu[c]) && std::isfinite(vv[c])) ? 1 : 0;
        }
        Mat1b vraster;
        gmRasterize(GH, GW, u_min, v_min, mvalid, Mat1b(), uc, ur, vraster);
        DS_valid[n] = gmDistTransform(vraster);
        if (real_overlap_native.empty() || !real_overlap_native.count(n)) {
            DS_active[n] = DS_valid[n].clone();
        } else {
            const Mat1b& ov = real_overlap_native.at(n);
            Mat1b active(s.H, s.W);
            for (int r = 0; r < s.H; ++r) {
                const uint8_t* mv = mvalid.ptr<uint8_t>(r);
                const uint8_t* op = ov.empty() ? nullptr : ov.ptr<uint8_t>(r);
                uint8_t* ap = active.ptr<uint8_t>(r);
                for (int c = 0; c < s.W; ++c) {
                    const bool is_overlap = op ? (op[c] != 0) : false;
                    ap[c] = (mv[c] && !is_overlap) ? 1 : 0;
                }
            }
            Mat1b araster;
            gmRasterize(GH, GW, u_min, v_min, mvalid, active, uc, ur, araster);
            DS_active[n] = gmDistTransform(araster);
        }
    }

    cv::Mat sum_active = cv::Mat::zeros(GH, GW, CV_64F);
    cv::Mat sum_valid  = cv::Mat::zeros(GH, GW, CV_64F);
    for (const auto& n : names) {
        sum_active += DS_active[n];
        sum_valid  += DS_valid[n];
    }
    std::unordered_map<std::string, cv::Mat> Wgrid;
    for (const auto& n : names) {
        cv::Mat W = cv::Mat::zeros(GH, GW, CV_64F);
        for (int r = 0; r < GH; ++r) {
            const double* sa = sum_active.ptr<double>(r);
            const double* sv = sum_valid.ptr<double>(r);
            const double* da = DS_active[n].ptr<double>(r);
            const double* dv = DS_valid[n].ptr<double>(r);
            double* wp = W.ptr<double>(r);
            for (int c = 0; c < GW; ++c) {
                if (sa[c] > 0.0)      wp[c] = da[c] / sa[c];
                else if (sv[c] > 0.0) wp[c] = dv[c] / sv[c];
                else                  wp[c] = 0.0;
            }
        }
        Wgrid[n] = W;
    }

    std::unordered_map<std::string, std::shared_ptr<cv::flann::Index>> trees;
    std::unordered_map<std::string, cv::Mat> tree_features, xyz_pts;
    for (const auto& n : names) {
        const GMSurface& s = surfs.at(n);
        const Mat1f& uc = uv.at(n).uc;
        const Mat1f& ur = uv.at(n).ur;
        std::vector<cv::Vec2f> pts; std::vector<cv::Vec3d> xs;
        pts.reserve(s.valid); xs.reserve(s.valid);
        for (int r = 0; r < s.H; ++r) {
            const uint8_t* mp = s.mask.ptr<uint8_t>(r);
            const float* uu = uc.ptr<float>(r);
            const float* vv = ur.ptr<float>(r);
            const float* xp = s.x.ptr<float>(r);
            const float* yp = s.y.ptr<float>(r);
            const float* zp = s.z.ptr<float>(r);
            for (int c = 0; c < s.W; ++c) {
                if (!mp[c]) continue;
                if (!std::isfinite(uu[c]) || !std::isfinite(vv[c])) continue;
                pts.emplace_back(uu[c], vv[c]);
                xs.emplace_back(xp[c], yp[c], zp[c]);
            }
        }
        cv::Mat feat((int)pts.size(), 2, CV_32F);
        for (size_t i = 0; i < pts.size(); ++i) {
            feat.at<float>((int)i, 0) = pts[i][0];
            feat.at<float>((int)i, 1) = pts[i][1];
        }
        cv::Mat xyz((int)xs.size(), 3, CV_64F);
        for (size_t i = 0; i < xs.size(); ++i) {
            xyz.at<double>((int)i, 0) = xs[i][0];
            xyz.at<double>((int)i, 1) = xs[i][1];
            xyz.at<double>((int)i, 2) = xs[i][2];
        }
        tree_features[n] = feat;
        xyz_pts[n] = xyz;
        trees[n] = std::make_shared<cv::flann::Index>(
            tree_features[n], cv::flann::KDTreeIndexParams(1));
    }

    auto idwQuery = [&](const std::string& src, const cv::Mat& q, cv::Mat& outXYZ) {
        const int Q = q.rows;
        const int K = idw_k;
        cv::Mat indices(Q, K, CV_32S);
        cv::Mat dists2(Q, K, CV_32F);
        trees[src]->knnSearch(q, indices, dists2, K, cv::flann::SearchParams(32));
        outXYZ.create(Q, 3, CV_64F);
        const cv::Mat& xyz = xyz_pts[src];
        for (int i = 0; i < Q; ++i) {
            double w[16], wsum = 0.0;
            const float* dr = dists2.ptr<float>(i);
            const int*   ir = indices.ptr<int>(i);
            for (int k = 0; k < K; ++k) {
                const double d = std::sqrt(std::max(0.0f, dr[k]));
                w[k] = 1.0 / (d + 1e-3);
                wsum += w[k];
            }
            for (int k = 0; k < K; ++k) w[k] /= wsum;
            double X=0,Y=0,Z=0;
            for (int k = 0; k < K; ++k) {
                const double* xp = xyz.ptr<double>(ir[k]);
                X += w[k]*xp[0]; Y += w[k]*xp[1]; Z += w[k]*xp[2];
            }
            double* op = outXYZ.ptr<double>(i);
            op[0] = X; op[1] = Y; op[2] = Z;
        }
    };

    std::unordered_map<std::string, GMBlend> out;
    const int Ns = (int)names.size();
    const bool irls_active = consensus_c > 0.0;

    for (const auto& tgt : names) {
        const GMSurface& s = surfs.at(tgt);
        const Mat1f& uc = uv.at(tgt).uc;
        const Mat1f& ur = uv.at(tgt).ur;
        const int Hm = s.H, Wm = s.W;
        const int Ntot = Hm * Wm;
        cv::Mat q(Ntot, 2, CV_32F);
        for (int r = 0; r < Hm; ++r) {
            const float* uu = uc.ptr<float>(r);
            const float* vv = ur.ptr<float>(r);
            for (int c = 0; c < Wm; ++c) {
                q.at<float>(r*Wm + c, 0) = uu[c];
                q.at<float>(r*Wm + c, 1) = vv[c];
            }
        }
        std::vector<std::vector<double>> ws_all(Ns, std::vector<double>(Ntot));
        std::vector<cv::Mat> sxyz_all(Ns);
        for (int s_i = 0; s_i < Ns; ++s_i) {
            const std::string& src = names[s_i];
            Mat1f wsamp = gmSampleBilinearReplicate(Wgrid[src], uc, ur, u_min, v_min);
            for (int r = 0; r < Hm; ++r) {
                const float* wp = wsamp.ptr<float>(r);
                for (int c = 0; c < Wm; ++c)
                    ws_all[s_i][r*Wm + c] = wp[c];
            }
            idwQuery(src, q, sxyz_all[s_i]);
        }
        std::vector<double> wsum(Ntot, 0.0);
        cv::Mat blended(Ntot, 3, CV_64F, cv::Scalar(0));
        for (int s_i = 0; s_i < Ns; ++s_i) {
            const auto& w = ws_all[s_i];
            const cv::Mat& sx = sxyz_all[s_i];
            for (int i = 0; i < Ntot; ++i) {
                wsum[i] += w[i];
                double* bp = blended.ptr<double>(i);
                const double* sp = sx.ptr<double>(i);
                bp[0] += w[i]*sp[0]; bp[1] += w[i]*sp[1]; bp[2] += w[i]*sp[2];
            }
        }
        for (int i = 0; i < Ntot; ++i) {
            if (wsum[i] > 1e-6) {
                double* bp = blended.ptr<double>(i);
                bp[0] /= wsum[i]; bp[1] /= wsum[i]; bp[2] /= wsum[i];
            }
        }
        if (irls_active) {
            int n_irls_cells=0, n_good=0, n_demoted=0;
            std::vector<double> deltas;
            deltas.reserve(Ntot/4);
            for (int i = 0; i < Ntot; ++i) {
                if (wsum[i] <= 1e-6) continue;
                int active_count = 0;
                for (int s_i = 0; s_i < Ns; ++s_i)
                    if (ws_all[s_i][i] > 1e-3) ++active_count;
                if (active_count < consensus_min_active) continue;
                ++n_irls_cells;
                const double* bp = blended.ptr<double>(i);
                double tukey[64]; double irls_w[64]; double irls_sum=0.0;
                for (int s_i = 0; s_i < Ns; ++s_i) {
                    const double* sp = sxyz_all[s_i].ptr<double>(i);
                    const double dx = sp[0]-bp[0], dy = sp[1]-bp[1], dz = sp[2]-bp[2];
                    const double r = std::sqrt(dx*dx+dy*dy+dz*dz);
                    const double u = r / consensus_c;
                    const double t = (u < 1.0) ? (1.0-u*u)*(1.0-u*u) : 0.0;
                    tukey[s_i] = t;
                    irls_w[s_i] = ws_all[s_i][i] * t;
                    irls_sum += irls_w[s_i];
                }
                if (irls_sum <= 1e-9) continue;
                double nx=0, ny=0, nz=0;
                for (int s_i = 0; s_i < Ns; ++s_i) {
                    const double* sp = sxyz_all[s_i].ptr<double>(i);
                    nx += irls_w[s_i]*sp[0]; ny += irls_w[s_i]*sp[1]; nz += irls_w[s_i]*sp[2];
                }
                nx /= irls_sum; ny /= irls_sum; nz /= irls_sum;
                ++n_good;
                for (int s_i = 0; s_i < Ns; ++s_i)
                    if (tukey[s_i] < 0.5 && ws_all[s_i][i] > 1e-3) ++n_demoted;
                double* bp_w = blended.ptr<double>(i);
                const double dlx = nx - bp_w[0];
                const double dly = ny - bp_w[1];
                const double dlz = nz - bp_w[2];
                deltas.push_back(std::sqrt(dlx*dlx+dly*dly+dlz*dlz));
                bp_w[0] = nx; bp_w[1] = ny; bp_w[2] = nz;
            }
            if (n_good > 0) {
                std::sort(deltas.begin(), deltas.end());
                const double p50 = deltas[deltas.size()/2];
                const double p95 = deltas[(size_t)(deltas.size()*0.95)];
                const double mx  = deltas.back();
                std::cout << "  " << tgt << " IRLS: " << n_good << "/" << n_irls_cells
                          << " 3+ cells re-blended  Δ p50="
                          << std::fixed << std::setprecision(2) << p50
                          << " p95=" << p95 << " max=" << mx
                          << "  sources_demoted=" << n_demoted << "\n";
                std::cout.unsetf(std::ios::fixed);
            } else if (n_irls_cells > 0) {
                std::cout << "  " << tgt << " IRLS: 0/" << n_irls_cells
                          << " 3+ cells (all sources outside Tukey c="
                          << consensus_c << ")\n";
            }
        }
        GMBlend bo;
        bo.X.create(Hm, Wm); bo.Y.create(Hm, Wm); bo.Z.create(Hm, Wm);
        for (int r = 0; r < Hm; ++r) {
            const uint8_t* mp = s.mask.ptr<uint8_t>(r);
            const float* uu = uc.ptr<float>(r);
            const float* vv = ur.ptr<float>(r);
            float* xp = bo.X.ptr<float>(r);
            float* yp = bo.Y.ptr<float>(r);
            float* zp = bo.Z.ptr<float>(r);
            for (int c = 0; c < Wm; ++c) {
                const int i = r*Wm + c;
                const bool ok = mp[c] && std::isfinite(uu[c]) && std::isfinite(vv[c]);
                const double* bp = blended.ptr<double>(i);
                xp[c] = ok ? (float)bp[0] : -1.0f;
                yp[c] = ok ? (float)bp[1] : -1.0f;
                zp[c] = ok ? (float)bp[2] : -1.0f;
            }
        }
        Mat1f own_w = gmSampleBilinearReplicate(Wgrid[tgt], uc, ur, u_min, v_min);
        std::vector<double> shifts; shifts.reserve(Ntot/4);
        for (int r = 0; r < Hm; ++r) {
            const uint8_t* mp = s.mask.ptr<uint8_t>(r);
            const float* uu = uc.ptr<float>(r);
            const float* vv = ur.ptr<float>(r);
            const float* ow = own_w.ptr<float>(r);
            const float* xp_old = s.x.ptr<float>(r);
            const float* yp_old = s.y.ptr<float>(r);
            const float* zp_old = s.z.ptr<float>(r);
            const float* xp_new = bo.X.ptr<float>(r);
            const float* yp_new = bo.Y.ptr<float>(r);
            const float* zp_new = bo.Z.ptr<float>(r);
            for (int c = 0; c < Wm; ++c) {
                if (!mp[c]) continue;
                if (!std::isfinite(uu[c]) || !std::isfinite(vv[c])) continue;
                if (!(ow[c] > 1e-3 && ow[c] < 1.0 - 1e-3)) continue;
                const double dx = xp_new[c]-xp_old[c];
                const double dy = yp_new[c]-yp_old[c];
                const double dz = zp_new[c]-zp_old[c];
                shifts.push_back(std::sqrt(dx*dx+dy*dy+dz*dz));
            }
        }
        if (!shifts.empty()) {
            std::sort(shifts.begin(), shifts.end());
            const double p50 = shifts[shifts.size()/2];
            const double p95 = shifts[(size_t)(shifts.size()*0.95)];
            const double mx  = shifts.back();
            std::cout << "  " << tgt << " overlap shift (vox): p50="
                      << std::fixed << std::setprecision(1) << p50
                      << "  p95=" << p95 << "  max=" << mx
                      << "  (cells=" << shifts.size() << ")\n";
            std::cout.unsetf(std::ios::fixed);
        } else {
            std::cout << "  " << tgt << ": no overlap cells\n";
        }
        out[tgt] = std::move(bo);
    }
    return out;
}

// -----------------------------------------------------------------------------
// Straighten stage: re-parameterize a freshly-merged tifxyz toward iso-theta
// columns + iso-z rows around the patch's umbilicus, blending the new UV with
// the current one per axis. Operates on the rasterized merge output, writes a
// new OBJ + invokes vc_obj2tifxyz_legacy a second time for the final tifxyz.
// -----------------------------------------------------------------------------

constexpr double kStnRapidThreshDeg = 30.0;
constexpr double kStnZRes           = 1.0;
constexpr double kStnSmoothSigma    = 20.0;

struct StnSurface {
    Mat1f x, y, z;
    Mat1b mask;
    int   H{0}, W{0}, valid{0};
};

StnSurface stnLoadTifxyz(const fs::path& p)
{
    auto qs = load_quad_from_tifxyz(p.string());
    if (!qs) throw std::runtime_error("failed to load tifxyz: " + p.string());
    const cv::Mat_<cv::Vec3f> pts = qs->rawPoints();
    StnSurface s;
    s.H = pts.rows;
    s.W = pts.cols;
    s.x.create(s.H, s.W);
    s.y.create(s.H, s.W);
    s.z.create(s.H, s.W);
    s.mask = Mat1b::zeros(s.H, s.W);
    int valid = 0;
    for (int r = 0; r < s.H; ++r) {
        const cv::Vec3f* pr = pts.ptr<cv::Vec3f>(r);
        for (int c = 0; c < s.W; ++c) {
            s.x(r, c) = pr[c][0];
            s.y(r, c) = pr[c][1];
            s.z(r, c) = pr[c][2];
            const bool ok = !(pr[c][0] == -1.0f && pr[c][1] == -1.0f && pr[c][2] == -1.0f);
            s.mask(r, c) = ok ? 1 : 0;
            if (ok) ++valid;
        }
    }
    s.valid = valid;
    return s;
}

// 1-D Gaussian filter, mode = "nearest", truncate = 4 (matches scipy).
void stnGaussianFilter1D(std::vector<double>& v, double sigma)
{
    if (sigma <= 0.0 || v.size() < 2) return;
    const int radius = (int)std::ceil(4.0 * sigma);
    std::vector<double> kernel((size_t)(2 * radius + 1));
    double s = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        const double w = std::exp(-0.5 * double(i * i) / (sigma * sigma));
        kernel[i + radius] = w;
        s += w;
    }
    for (double& w : kernel) w /= s;
    const int n = (int)v.size();
    std::vector<double> out((size_t)n);
    for (int i = 0; i < n; ++i) {
        double acc = 0.0;
        for (int k = -radius; k <= radius; ++k) {
            int j = i + k;
            if (j < 0) j = 0;
            else if (j >= n) j = n - 1;
            acc += kernel[(size_t)(k + radius)] * v[(size_t)j];
        }
        out[(size_t)i] = acc;
    }
    v = std::move(out);
}

// Compute U(z) = mean (x, y) per z bin (z_res=1.0), linear-extrapolate empty
// bins, gaussian-smooth with sigma=20 along z.
void stnComputeUmbilicus(const StnSurface& s,
                         std::vector<double>& z_grid,
                         std::vector<double>& U_x,
                         std::vector<double>& U_y)
{
    double z_min = std::numeric_limits<double>::infinity();
    double z_max = -std::numeric_limits<double>::infinity();
    for (int r = 0; r < s.H; ++r) {
        for (int c = 0; c < s.W; ++c) {
            if (!s.mask(r, c)) continue;
            const double zv = s.z(r, c);
            if (zv < z_min) z_min = zv;
            if (zv > z_max) z_max = zv;
        }
    }
    if (!std::isfinite(z_min) || !std::isfinite(z_max))
        throw std::runtime_error("straighten: no valid cells in input");
    const int n = (int)std::ceil((z_max - z_min) / kStnZRes) + 1;
    U_x.assign((size_t)n, 0.0);
    U_y.assign((size_t)n, 0.0);
    std::vector<double> W((size_t)n, 0.0);
    for (int r = 0; r < s.H; ++r) {
        for (int c = 0; c < s.W; ++c) {
            if (!s.mask(r, c)) continue;
            int idx = (int)((s.z(r, c) - z_min) / kStnZRes);
            if (idx < 0) idx = 0;
            if (idx >= n) idx = n - 1;
            U_x[(size_t)idx] += s.x(r, c);
            U_y[(size_t)idx] += s.y(r, c);
            W  [(size_t)idx] += 1.0;
        }
    }
    for (int i = 0; i < n; ++i) {
        if (W[(size_t)i] > 0) {
            U_x[(size_t)i] /= W[(size_t)i];
            U_y[(size_t)i] /= W[(size_t)i];
        }
    }
    auto findNext = [&](int from) {
        for (int i = from; i < n; ++i) if (W[(size_t)i] > 0) return i;
        return -1;
    };
    auto findPrev = [&](int from) {
        for (int i = from; i >= 0; --i) if (W[(size_t)i] > 0) return i;
        return -1;
    };
    for (int i = 0; i < n; ++i) {
        if (W[(size_t)i] > 0) continue;
        const int prev = findPrev(i - 1);
        const int next = findNext(i + 1);
        if (prev >= 0 && next >= 0) {
            const double t = double(i - prev) / double(next - prev);
            U_x[(size_t)i] = (1 - t) * U_x[(size_t)prev] + t * U_x[(size_t)next];
            U_y[(size_t)i] = (1 - t) * U_y[(size_t)prev] + t * U_y[(size_t)next];
        } else if (prev >= 0) {
            const int prev2 = findPrev(prev - 1);
            if (prev2 >= 0) {
                const double inv = 1.0 / double(prev - prev2);
                const double dx = (U_x[(size_t)prev] - U_x[(size_t)prev2]) * inv;
                const double dy = (U_y[(size_t)prev] - U_y[(size_t)prev2]) * inv;
                U_x[(size_t)i] = U_x[(size_t)prev] + dx * double(i - prev);
                U_y[(size_t)i] = U_y[(size_t)prev] + dy * double(i - prev);
            } else {
                U_x[(size_t)i] = U_x[(size_t)prev];
                U_y[(size_t)i] = U_y[(size_t)prev];
            }
        } else if (next >= 0) {
            const int next2 = findNext(next + 1);
            if (next2 >= 0) {
                const double inv = 1.0 / double(next2 - next);
                const double dx = (U_x[(size_t)next2] - U_x[(size_t)next]) * inv;
                const double dy = (U_y[(size_t)next2] - U_y[(size_t)next]) * inv;
                U_x[(size_t)i] = U_x[(size_t)next] + dx * double(i - next);
                U_y[(size_t)i] = U_y[(size_t)next] + dy * double(i - next);
            } else {
                U_x[(size_t)i] = U_x[(size_t)next];
                U_y[(size_t)i] = U_y[(size_t)next];
            }
        }
    }
    stnGaussianFilter1D(U_x, kStnSmoothSigma);
    stnGaussianFilter1D(U_y, kStnSmoothSigma);
    z_grid.resize((size_t)n);
    for (int i = 0; i < n; ++i) z_grid[(size_t)i] = z_min + i * kStnZRes;
}

double stnInterpLinear(const std::vector<double>& xg,
                       const std::vector<double>& yv,
                       double x)
{
    if (xg.empty()) return 0.0;
    if (x <= xg.front()) return yv.front();
    if (x >= xg.back())  return yv.back();
    auto it = std::upper_bound(xg.begin(), xg.end(), x);
    const int hi = (int)(it - xg.begin());
    const int lo = hi - 1;
    const double t = (x - xg[(size_t)lo]) / (xg[(size_t)hi] - xg[(size_t)lo]);
    return (1.0 - t) * yv[(size_t)lo] + t * yv[(size_t)hi];
}

double stnShortAngDiff(double a, double b)
{
    double d = std::fmod(b - a + M_PI, 2.0 * M_PI);
    if (d < 0) d += 2.0 * M_PI;
    return d - M_PI;
}

cv::Mat_<double> stnBfsUnwrap(const cv::Mat_<double>& theta,
                              const Mat1b& mask,
                              double thresholdRad,
                              int& numRejected,
                              std::pair<int, int>& seedOut)
{
    const int H = theta.rows, W = theta.cols;
    cv::Mat_<double> th_cont(H, W, std::numeric_limits<double>::quiet_NaN());
    std::vector<int> vr, vc;
    vr.reserve((size_t)H * (size_t)W / 4u);
    vc.reserve((size_t)H * (size_t)W / 4u);
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            if (mask(r, c)) { vr.push_back(r); vc.push_back(c); }
    if (vr.empty()) {
        numRejected = 0;
        seedOut = {-1, -1};
        return th_cont;
    }
    std::vector<int> vrs = vr, vcs = vc;
    std::sort(vrs.begin(), vrs.end());
    std::sort(vcs.begin(), vcs.end());
    int mid_r = vrs[vrs.size() / 2];
    int mid_c = vcs[vcs.size() / 2];
    if (!mask(mid_r, mid_c)) {
        long long bestD = std::numeric_limits<long long>::max();
        int bR = mid_r, bC = mid_c;
        for (size_t i = 0; i < vr.size(); ++i) {
            const long long dr = vr[i] - mid_r;
            const long long dc = vc[i] - mid_c;
            const long long d  = dr * dr + dc * dc;
            if (d < bestD) { bestD = d; bR = vr[i]; bC = vc[i]; }
        }
        mid_r = bR; mid_c = bC;
    }
    th_cont(mid_r, mid_c) = theta(mid_r, mid_c);
    std::deque<std::pair<int, int>> q;
    q.emplace_back(mid_r, mid_c);
    int rej = 0;
    static const int dr[4] = {-1,  1,  0, 0};
    static const int dc[4] = { 0,  0, -1, 1};
    while (!q.empty()) {
        const auto [r, c] = q.front(); q.pop_front();
        const double tr_c = th_cont(r, c);
        for (int k = 0; k < 4; ++k) {
            const int nr = r + dr[k], nc = c + dc[k];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            if (!mask(nr, nc) || !std::isnan(th_cont(nr, nc))) continue;
            const double d = stnShortAngDiff(theta(r, c), theta(nr, nc));
            if (std::abs(d) > thresholdRad) { ++rej; continue; }
            th_cont(nr, nc) = tr_c + d;
            q.emplace_back(nr, nc);
        }
    }
    numRejected = rej;
    seedOut = {mid_r, mid_c};
    return th_cont;
}

double stnPercentile(std::vector<double>& v, double p)
{
    if (v.empty()) return 0.0;
    const size_t k = (size_t)std::floor(p / 100.0 * double(v.size() - 1));
    std::nth_element(v.begin(), v.begin() + (long)k, v.end());
    return v[k];
}

double stnMaxAbs(const std::vector<double>& v)
{
    double m = 0;
    for (double x : v) { const double a = std::abs(x); if (a > m) m = a; }
    return m;
}

void stnWriteObj(const fs::path& path,
                 const StnSurface& s,
                 const cv::Mat_<double>& u_new,
                 const cv::Mat_<double>& v_new,
                 double strengthU, double strengthV)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot write OBJ: " + path.string());
    f << "# vc_merge_tifxyz straighten (alpha_u=" << strengthU
      << ", alpha_v=" << strengthV
      << ", rapid-reject=" << kStnRapidThreshDeg << " deg)\n";

    const int H = s.H, W = s.W;
    cv::Mat_<int> idx(H, W, -1);
    int n_v = 0;
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            if (s.mask(r, c)) idx(r, c) = n_v++;

    std::ostringstream vbuf, vtbuf, fbuf;
    vbuf  << std::fixed << std::setprecision(4);
    vtbuf << std::fixed << std::setprecision(4);
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            if (idx(r, c) < 0) continue;
            vbuf  << "v "  << s.x(r, c) << " " << s.y(r, c) << " " << s.z(r, c) << "\n";
            vtbuf << "vt " << u_new(r, c) << " " << v_new(r, c) << "\n";
        }
    }
    int n_q = 0;
    for (int r = 0; r < H - 1; ++r) {
        for (int c = 0; c < W - 1; ++c) {
            const int a = idx(r,     c);
            const int b = idx(r,     c + 1);
            const int cc= idx(r + 1, c);
            const int d = idx(r + 1, c + 1);
            if (a < 0 || b < 0 || cc < 0 || d < 0) continue;
            fbuf << "f " << (a + 1)  << "/" << (a + 1)
                 << " "  << (b + 1)  << "/" << (b + 1)
                 << " "  << (d + 1)  << "/" << (d + 1) << "\n";
            fbuf << "f " << (a + 1)  << "/" << (a + 1)
                 << " "  << (d + 1)  << "/" << (d + 1)
                 << " "  << (cc + 1) << "/" << (cc + 1) << "\n";
            ++n_q;
        }
    }
    f << vbuf.str() << vtbuf.str() << fbuf.str();
    std::cout << "writing OBJ: " << n_v << " verts, " << (2 * n_q)
              << " tris -> " << path << "\n";
}

void stnPatchMetaUuid(const fs::path& tifxyz_dir)
{
    fs::path meta_p = tifxyz_dir / "meta.json";
    if (!fs::exists(meta_p)) return;
    json m;
    {
        std::ifstream f(meta_p);
        f >> m;
    }
    m["uuid"] = tifxyz_dir.filename().string();
    {
        std::ofstream f(meta_p);
        f << m.dump();
    }
}

int rasterizeObjToTifxyz(const fs::path& obj2tifxyz,
                         const fs::path& obj,
                         const fs::path& output,
                         int step_size)
{
    // vc_obj2tifxyz_legacy refuses if `output` already exists, so we wipe it.
    // Callers that want the OBJ inside `output` must write the OBJ to a
    // sibling temp path, rasterize, then move the OBJ in afterwards.
    if (fs::exists(output)) fs::remove_all(output);
    std::ostringstream cmd;
    cmd << "\"" << obj2tifxyz.string() << "\" "
        << "\"" << obj.string() << "\" "
        << "\"" << output.string() << "\" "
        << step_size;
    const int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
        std::cerr << "vc_obj2tifxyz_legacy failed (rc=" << rc << ")\n";
        return 1;
    }
    stnPatchMetaUuid(output);
    return 0;
}

void reportTifxyzShape(const fs::path& dir, const std::string& label)
{
    cv::Mat ox = cv::imread((dir / "x.tif").string(), cv::IMREAD_UNCHANGED);
    if (ox.empty() || ox.depth() != CV_32F) return;
    int valid = 0;
    for (int r = 0; r < ox.rows; ++r) {
        const float* p = ox.ptr<float>(r);
        for (int c = 0; c < ox.cols; ++c) if (p[c] != -1.0f) ++valid;
    }
    std::cout << label << ": (" << ox.rows << ", " << ox.cols
              << "), valid=" << valid << "\n";
}

fs::path resolveObj2Tifxyz(const std::string& override_path, const char* argv0)
{
    if (!override_path.empty()) return override_path;
    try {
        const fs::path self = fs::canonical(argv0);
        const fs::path adj  = self.parent_path() / "vc_obj2tifxyz_legacy";
        if (fs::exists(adj)) return adj;
    } catch (...) { /* fall through to PATH lookup */ }
    return fs::path("vc_obj2tifxyz_legacy");
}

int stnRun(const fs::path& surf_in,
           const fs::path& output,
           const fs::path& obj_out,
           const fs::path& obj2tifxyz,
           int step_size,
           double strengthU,
           double strengthV)
{
    std::cout << "loading " << surf_in << "\n";
    const StnSurface s = stnLoadTifxyz(surf_in);
    std::cout << "  " << s.H << "x" << s.W << ", valid=" << s.valid << "\n";

    std::cout << "umbilicus\n";
    std::vector<double> z_grid, U_x, U_y;
    stnComputeUmbilicus(s, z_grid, U_x, U_y);

    cv::Mat_<double> theta(s.H, s.W);
    std::vector<double> r_valid;
    r_valid.reserve((size_t)s.valid);
    for (int r = 0; r < s.H; ++r) {
        for (int c = 0; c < s.W; ++c) {
            const double zv = s.z(r, c);
            const double ux = stnInterpLinear(z_grid, U_x, zv);
            const double uy = stnInterpLinear(z_grid, U_y, zv);
            const double dx = s.x(r, c) - ux;
            const double dy = s.y(r, c) - uy;
            theta(r, c) = std::atan2(dy, dx);
            if (s.mask(r, c)) r_valid.push_back(std::hypot(dx, dy));
        }
    }
    {
        std::vector<double> tmp = r_valid;
        std::cout << "  r_med=" << std::fixed << std::setprecision(1)
                  << stnPercentile(tmp, 50) << "\n";
        std::cout.unsetf(std::ios::fixed);
    }

    std::cout << "BFS unwrap (reject edges > "
              << std::fixed << std::setprecision(1) << kStnRapidThreshDeg
              << " deg)\n";
    std::cout.unsetf(std::ios::fixed);
    int rej = 0;
    std::pair<int, int> seed{-1, -1};
    cv::Mat_<double> th_cont = stnBfsUnwrap(theta, s.mask,
                                            kStnRapidThreshDeg * M_PI / 180.0,
                                            rej, seed);
    Mat1b reachable(s.H, s.W, (uint8_t)0);
    int reachableCount = 0;
    double tmin =  std::numeric_limits<double>::infinity();
    double tmax = -std::numeric_limits<double>::infinity();
    for (int r = 0; r < s.H; ++r) {
        for (int c = 0; c < s.W; ++c) {
            if (!s.mask(r, c) || !std::isfinite(th_cont(r, c))) continue;
            reachable(r, c) = 1;
            ++reachableCount;
            const double t = th_cont(r, c) * 180.0 / M_PI;
            if (t < tmin) tmin = t;
            if (t > tmax) tmax = t;
        }
    }
    std::cout << "  seed=(" << seed.first << ", " << seed.second
              << "), reachable=" << reachableCount << "/" << s.valid
              << ", rejected=" << rej << "\n"
              << "  theta_cont (deg): min=" << std::fixed << std::setprecision(1)
              << tmin << ", max=" << tmax << ", span=" << (tmax - tmin) << "\n";
    std::cout.unsetf(std::ios::fixed);

    Eigen::MatrixXd A((Eigen::Index)reachableCount, 3);
    Eigen::VectorXd Bu((Eigen::Index)reachableCount);
    Eigen::VectorXd Bv((Eigen::Index)reachableCount);
    {
        Eigen::Index i = 0;
        for (int r = 0; r < s.H; ++r) {
            for (int c = 0; c < s.W; ++c) {
                if (!reachable(r, c)) continue;
                A(i, 0) = th_cont(r, c);
                A(i, 1) = s.z(r, c);
                A(i, 2) = 1.0;
                Bu(i) = double(c) * step_size;
                Bv(i) = double(r) * step_size;
                ++i;
            }
        }
    }
    const Eigen::Vector3d ucoef = A.colPivHouseholderQr().solve(Bu);
    const Eigen::Vector3d vcoef = A.colPivHouseholderQr().solve(Bv);
    const double a = ucoef(0), b = ucoef(1), cc = ucoef(2);
    const double d = vcoef(0), e = vcoef(1), fcoef = vcoef(2);
    std::cout << "  affine (theta,z)->(u,v): "
              << "u = " << std::showpos << std::fixed << std::setprecision(2) << a
              << "*theta_rad " << std::setprecision(3) << b
              << "*z " << std::setprecision(1) << cc << " | "
              << "v = " << std::setprecision(2) << d
              << "*theta_rad " << std::setprecision(3) << e
              << "*z " << std::setprecision(1) << fcoef
              << std::noshowpos << "\n";
    std::cout.unsetf(std::ios::fixed);

    {
        std::vector<double> resu, resv;
        resu.reserve((size_t)reachableCount);
        resv.reserve((size_t)reachableCount);
        for (int r = 0; r < s.H; ++r) {
            for (int c = 0; c < s.W; ++c) {
                if (!reachable(r, c)) continue;
                const double pu = a * th_cont(r, c) + b * s.z(r, c) + cc;
                const double pv = d * th_cont(r, c) + e * s.z(r, c) + fcoef;
                resu.push_back(std::abs(double(c) * step_size - pu));
                resv.push_back(std::abs(double(r) * step_size - pv));
            }
        }
        std::vector<double> cu = resu, cv2 = resv;
        const double u50 = stnPercentile(cu,  50);
        const double u95 = stnPercentile(cu,  95);
        const double umax = stnMaxAbs(resu);
        const double v50 = stnPercentile(cv2, 50);
        const double v95 = stnPercentile(cv2, 95);
        const double vmax = stnMaxAbs(resv);
        std::cout << "  affine residual (vox): u p50=" << std::fixed << std::setprecision(1) << u50
                  << " p95=" << u95 << " max=" << umax << "\n"
                  << "  affine residual (vox): v p50=" << v50
                  << " p95=" << v95 << " max=" << vmax << "\n";
        std::cout.unsetf(std::ios::fixed);
    }

    cv::Mat_<double> u_new(s.H, s.W), v_new(s.H, s.W);
    std::vector<double> du_abs, dv_abs;
    du_abs.reserve((size_t)s.valid);
    dv_abs.reserve((size_t)s.valid);
    for (int r = 0; r < s.H; ++r) {
        for (int c = 0; c < s.W; ++c) {
            const double u_cur = double(c) * step_size;
            const double v_cur = double(r) * step_size;
            if (reachable(r, c)) {
                const double u_tgt = a * th_cont(r, c) + b * s.z(r, c) + cc;
                const double v_tgt = d * th_cont(r, c) + e * s.z(r, c) + fcoef;
                u_new(r, c) = (1.0 - strengthU) * u_cur + strengthU * u_tgt;
                v_new(r, c) = (1.0 - strengthV) * v_cur + strengthV * v_tgt;
            } else {
                u_new(r, c) = u_cur;
                v_new(r, c) = v_cur;
            }
            if (s.mask(r, c)) {
                du_abs.push_back(std::abs(u_new(r, c) - u_cur));
                dv_abs.push_back(std::abs(v_new(r, c) - v_cur));
            }
        }
    }
    {
        std::vector<double> cu = du_abs, cv2 = dv_abs;
        const double u50 = stnPercentile(cu,  50);
        const double u95 = stnPercentile(cu,  95);
        const double umax = stnMaxAbs(du_abs);
        const double v50 = stnPercentile(cv2, 50);
        const double v95 = stnPercentile(cv2, 95);
        const double vmax = stnMaxAbs(dv_abs);
        std::cout << "UV shift (vox): u p50=" << std::fixed << std::setprecision(1) << u50
                  << " p95=" << u95 << " max=" << umax << "\n"
                  << "UV shift (vox): v p50=" << v50
                  << " p95=" << v95 << " max=" << vmax << "\n";
        std::cout.unsetf(std::ios::fixed);
    }

    // obj_out is the FINAL destination (inside `output`). We have to write the
    // OBJ to a sibling temp first, because vc_obj2tifxyz_legacy refuses an
    // existing `output` dir — then move the OBJ inside once rasterization
    // succeeds.
    const fs::path obj_tmp = output.parent_path() / obj_out.filename();
    stnWriteObj(obj_tmp, s, u_new, v_new, strengthU, strengthV);

    std::cout << "rasterizing -> " << output << "\n";
    if (const int rc = rasterizeObjToTifxyz(obj2tifxyz, obj_tmp, output, step_size); rc != 0)
        return rc;
    fs::rename(obj_tmp, obj_out);
    reportTifxyzShape(output, "output");
    std::cout << "wrote: " << output << "\n";
    return 0;
}

std::pair<int,int> gmEmitMesh(const Mat1f& X, const Mat1f& Y, const Mat1f& Z,
                              const Mat1b& mask,
                              const Mat1f& uc, const Mat1f& ur,
                              std::ostream& obj,
                              int v_offset, int vt_offset, int step_size)
{
    const int H = X.rows, W = X.cols;
    std::vector<int64_t> idx((size_t)H * W, -1);
    auto IDX = [&](int r, int c) -> int64_t& { return idx[(size_t)r * W + c]; };
    int n_v = 0;
    for (int r = 0; r < H; ++r) {
        const uint8_t* mp = mask.ptr<uint8_t>(r);
        const float* uu = uc.ptr<float>(r);
        const float* vv = ur.ptr<float>(r);
        for (int c = 0; c < W; ++c) {
            if (!mp[c]) continue;
            if (!std::isfinite(uu[c]) || !std::isfinite(vv[c])) continue;
            IDX(r, c) = (int64_t)n_v + v_offset;
            ++n_v;
        }
    }
    std::ostringstream vbuf, vtbuf;
    vbuf  << std::fixed << std::setprecision(4);
    vtbuf << std::fixed << std::setprecision(4);
    for (int r = 0; r < H; ++r) {
        const float* xp = X.ptr<float>(r);
        const float* yp = Y.ptr<float>(r);
        const float* zp = Z.ptr<float>(r);
        const float* uu = uc.ptr<float>(r);
        const float* vv = ur.ptr<float>(r);
        for (int c = 0; c < W; ++c) {
            if (IDX(r, c) < 0) continue;
            vbuf  << "v " << xp[c] << " " << yp[c] << " " << zp[c] << "\n";
            vtbuf << "vt " << (double)uu[c] * step_size
                  << " " << (double)vv[c] * step_size << "\n";
        }
    }
    obj << vbuf.str();
    obj << vtbuf.str();
    int n_q = 0;
    std::ostringstream fbuf;
    auto vidx  = [&](int64_t a){ return a + 1; };
    auto vtidx = [&](int64_t a){ return (a - v_offset) + vt_offset + 1; };
    for (int r = 0; r < H - 1; ++r) {
        for (int c = 0; c < W - 1; ++c) {
            const int64_t a = IDX(r, c);
            const int64_t b = IDX(r, c+1);
            const int64_t cc= IDX(r+1, c);
            const int64_t d = IDX(r+1, c+1);
            if (a < 0 || b < 0 || cc < 0 || d < 0) continue;
            fbuf << "f " << vidx(a) << "/" << vtidx(a)
                 << " "  << vidx(b) << "/" << vtidx(b)
                 << " "  << vidx(d) << "/" << vtidx(d) << "\n";
            fbuf << "f " << vidx(a) << "/" << vtidx(a)
                 << " "  << vidx(d) << "/" << vtidx(d)
                 << " "  << vidx(cc) << "/" << vtidx(cc) << "\n";
            ++n_q;
        }
    }
    obj << fbuf.str();
    return {n_v, 2 * n_q};
}

// Scan paths_dir for any directory whose name is <base_final>, <base_pre>, or
// either base followed by "_v<digits>". Returns "" if nothing matches, else
// "_v<max+1>" (bare names count as v0). Used to bump both output dirs to the
// same version when either side already exists from a prior run.
std::string gmPickVersionSuffix(const fs::path& paths_dir,
                                const std::string& base_final,
                                const std::string& base_pre)
{
    if (!fs::is_directory(paths_dir)) return "";
    int max_v = -1;
    auto consider = [&](const std::string& name, const std::string& base) {
        if (name == base) {
            if (max_v < 0) max_v = 0;
            return;
        }
        const std::string prefix = base + "_v";
        if (name.size() <= prefix.size()) return;
        if (name.compare(0, prefix.size(), prefix) != 0) return;
        const std::string tail = name.substr(prefix.size());
        for (char c : tail)
            if (!std::isdigit(static_cast<unsigned char>(c))) return;
        try {
            int v = std::stoi(tail);
            if (v > max_v) max_v = v;
        } catch (...) {}
    };
    for (const auto& entry : fs::directory_iterator(paths_dir)) {
        if (!entry.is_directory()) continue;
        const std::string name = entry.path().filename().string();
        consider(name, base_final);
        consider(name, base_pre);
    }
    if (max_v < 0) return "";
    return "_v" + std::to_string(max_v + 1);
}

// -----------------------------------------------------------------------------
// Top-level driver: load surfaces, run pairwise per edge in-memory, run all
// global stages, emit OBJ + invoke vc_obj2tifxyz_legacy. Writes one summary
// JSON. Returns 0 on success.
// -----------------------------------------------------------------------------

int gmRunGlobalMerge(const fs::path& merge_path,
                     const fs::path& obj2tifxyz,
                     GMConfig cfg)
{
    cfg.paths_dir = merge_path.parent_path() / "paths";
    if (!fs::is_directory(cfg.paths_dir))
        throw std::runtime_error("expected " + cfg.paths_dir.string() +
            " (sibling of " + merge_path.string() + ") to be a directory");

    std::cout << "[0/7] grid -> surfaces+edges from " << merge_path << "\n";
    gmResolveGrid(merge_path, cfg.paths_dir, cfg.surfaces, cfg.edges);
    std::cout << "  " << cfg.surfaces.size() << " surfaces, "
              << cfg.edges.size() << " auto-derived edges\n";
    gmCheckConnected(cfg.surfaces, cfg.edges);

    // Output dir name is derived from the alphabetically-first surface in the
    // resolved grid. <alpha_first>_merged is the final tifxyz; the pre-
    // straighten intermediate is <alpha_first>_merged_prestraightened. Both
    // land under paths_dir alongside the input surfaces. If either of those
    // bases (or any prior _vN of either) already exists, both dirs are bumped
    // to the same _v(max+1) suffix so the lineage stays paired.
    std::string alpha_first = cfg.surfaces.front().name;
    for (const auto& s : cfg.surfaces)
        if (s.name < alpha_first) alpha_first = s.name;
    const std::string base_final = alpha_first + "_merged";
    const std::string base_pre   = alpha_first + "_merged_prestraightened";
    const std::string vsuffix    = gmPickVersionSuffix(cfg.paths_dir, base_final, base_pre);
    const std::string final_name = base_final + vsuffix;
    const std::string pre_name   = base_pre   + vsuffix;
    const fs::path output_dir = cfg.paths_dir / final_name;
    const fs::path pre_dir    = cfg.paths_dir / pre_name;
    // OBJ + summary live INSIDE their respective tifxyz dirs.
    const fs::path obj_out      = output_dir / (final_name + ".obj");
    const fs::path pre_obj      = pre_dir    / (pre_name + ".obj");
    const fs::path summary_path = output_dir / (final_name + "_summary.json");
    std::cout << "  output: " << output_dir << "\n"
              << "  pre-straighten: " << pre_dir << "\n";

    std::cout << "[1/6] loading surfaces" << std::endl;
    std::unordered_map<std::string, GMSurface> surfs;
    std::vector<std::string> names;
    names.reserve(cfg.surfaces.size());
    for (const auto& spec : cfg.surfaces) {
        GMSurface s = gmLoadSurface(spec);
        std::cout << "  " << s.name << ": shape=(" << s.H << "," << s.W
                  << "), valid=" << s.valid << "\n";
        names.push_back(s.name);
        surfs.emplace(s.name, std::move(s));
    }

    std::string ref = cfg.ref;
    if (ref.empty()) {
        int best = -1;
        for (const auto& n : names)
            if (surfs.at(n).valid > best) { best = surfs.at(n).valid; ref = n; }
    } else if (!surfs.count(ref)) {
        std::cerr << "ref '" << ref << "' is not in surfaces list\n";
        return 2;
    }
    std::cout << "  reference surface: " << ref
              << " (valid=" << surfs.at(ref).valid << ")\n";

    // Build a single SurfacePatchIndex over all surfaces — this lets each
    // edge's pairwise pass query overlaps without rebuilding.
    SurfacePatchIndex patchIndex;
    {
        std::vector<SurfacePatchIndex::SurfacePtr> all;
        all.reserve(names.size());
        for (const auto& n : names) all.push_back(surfs.at(n).qs);
        std::cout << "Building SurfacePatchIndex over " << all.size()
                  << " surfaces..." << std::endl;
        patchIndex.rebuild(all);
    }

    std::cout << "[2/6] per-edge overlap+anchors+RANSAC similarity\n";
    std::vector<GMEdgeRun> edge_runs;
    edge_runs.reserve(cfg.edges.size());
    json edges_json = json::array();
    std::unordered_map<std::string, Mat1b> real_overlap_native;
    for (const auto& n : names)
        real_overlap_native[n] = Mat1b::zeros(surfs.at(n).H, surfs.at(n).W);
    int total_inliers = 0;
    std::vector<GMEdgeSpec> dropped_edges;
    for (const auto& e : cfg.edges) {
        if (!surfs.count(e.a) || !surfs.count(e.b))
            throw std::runtime_error("global mode: unknown surface in edge "
                                     + e.a + "<->" + e.b);
        std::cout << "  " << e.a << "<->" << e.b << ":\n";
        try {
            EdgeAnchors pr = gmComputeEdgeAnchors(surfs.at(e.a).qs,
                                                  surfs.at(e.b).qs,
                                                  e.a, e.b, patchIndex, cfg);

            // Convert Anchor list → cv::Vec2d (col, row) pairs in cell coords.
            std::vector<cv::Vec2d> pA, pB;
            pA.reserve(pr.anchors.size());
            pB.reserve(pr.anchors.size());
            for (const auto& a : pr.anchors) {
                pA.emplace_back((double)a.a_col, (double)a.a_row);
                pB.emplace_back((double)a.b_col, (double)a.b_row);
            }

            GMRansac R = gmRansacSimilarity(pA, pB,
                cfg.ransac_iters, cfg.ransac_min_thresh, cfg.ransac_max_thresh,
                cfg.ransac_mad_k, cfg.ransac_seed);

            GMEdgeRun er;
            er.a = e.a; er.b = e.b;
            er.M_pair = R.M; er.sigma_in = R.sigma_in; er.thresh = R.thresh;
            er.n_in = R.n_in; er.n_total = (int)pA.size();
            er.pA.reserve(R.n_in); er.pB.reserve(R.n_in);
            for (size_t i = 0; i < pA.size(); ++i)
                if (R.inlier[i]) { er.pA.push_back(pA[i]); er.pB.push_back(pB[i]); }

            // Accumulate real-overlap masks per surface.
            if (!pr.realA.empty() && pr.realA.size() == real_overlap_native[e.a].size()) {
                int oA = 0;
                Mat1b& dst = real_overlap_native[e.a];
                for (int r = 0; r < pr.realA.rows; ++r)
                    for (int c = 0; c < pr.realA.cols; ++c)
                        if (pr.realA(r,c)) { dst(r,c) = 1; ++oA; }
                er.real_overlap_a = oA;
            }
            if (!pr.realB.empty() && pr.realB.size() == real_overlap_native[e.b].size()) {
                int oB = 0;
                Mat1b& dst = real_overlap_native[e.b];
                for (int r = 0; r < pr.realB.rows; ++r)
                    for (int c = 0; c < pr.realB.cols; ++c)
                        if (pr.realB(r,c)) { dst(r,c) = 1; ++oB; }
                er.real_overlap_b = oB;
            }

            const double pair_sc = gmSimilarityScale(er.M_pair);
            std::cout << "    inliers=" << er.n_in << "/" << er.n_total
                      << "  (thresh=" << std::fixed << std::setprecision(2) << er.thresh
                      << ")  pair scale=" << std::fixed << std::setprecision(4) << pair_sc
                      << "  sigma_in=" << std::fixed << std::setprecision(2) << er.sigma_in
                      << "  real-overlap A=" << er.real_overlap_a
                      << " B=" << er.real_overlap_b << "\n";
            std::cout.unsetf(std::ios::fixed);
            total_inliers += er.n_in;

            json ej = pr.edge_json;
            ej["ransac_inliers"] = er.n_in;
            ej["ransac_total"]   = er.n_total;
            ej["ransac_thresh"]  = er.thresh;
            ej["ransac_sigma_in"]= er.sigma_in;
            ej["pair_scale"]     = pair_sc;
            ej["real_overlap_A"] = er.real_overlap_a;
            ej["real_overlap_B"] = er.real_overlap_b;
            edges_json.push_back(std::move(ej));
            edge_runs.push_back(std::move(er));
        } catch (const std::exception& ex) {
            std::cout << "    WARN: " << ex.what() << " — dropping edge\n";
            dropped_edges.push_back(e);
        }
    }
    if (!dropped_edges.empty()) {
        std::cout << "  dropped " << dropped_edges.size() << " edge(s) with no "
                                                             "valid alignment:\n";
        for (const auto& e : dropped_edges)
            std::cout << "    " << e.a << "<->" << e.b << "\n";
        std::vector<GMEdgeSpec> kept;
        kept.reserve(edge_runs.size());
        for (const auto& er : edge_runs) kept.push_back({er.a, er.b});
        gmCheckConnected(cfg.surfaces, kept);
    }

    std::cout << "[3/6] joint affine fit (ref=" << ref
              << ", " << total_inliers << " inlier anchors)\n";
    std::unordered_map<std::string, cv::Matx23d> M_dict =
        gmJointAffine(names, edge_runs, ref, kRidgeLambda);
    json joint_json = json::array();
    for (const auto& n : names) {
        const auto& M = M_dict.at(n);
        auto [sx, sy] = gmAffineScales(M);
        const double aniso = std::max(sx, sy) / std::max(std::min(sx, sy), 1e-9);
        std::cout << "  " << n
                  << ": sx=" << std::fixed << std::setprecision(4) << sx
                  << " sy="  << std::fixed << std::setprecision(4) << sy
                  << " aniso=" << std::fixed << std::setprecision(3) << aniso
                  << "  t=[" << std::showpos << std::fixed << std::setprecision(2)
                  << M(0,2) << ", " << M(1,2) << "]" << std::noshowpos << "\n";
        std::cout.unsetf(std::ios::fixed);
        joint_json.push_back({
            {"surface", n}, {"sx", sx}, {"sy", sy}, {"aniso", aniso},
            {"t", {M(0,2), M(1,2)}},
            {"M", {{M(0,0), M(0,1), M(0,2)}, {M(1,0), M(1,1), M(1,2)}}},
        });
    }

    std::cout << "[4/6] per-surface RBF on union of incident midpoint residuals\n";
    auto warps = gmBuildSurfaceWarps(names, edge_runs, M_dict);
    json rbf_json = json::array();
    for (const auto& n : names) {
        const auto& w = warps[n];
        std::cout << "  " << n << ": " << w.n_anchors << " anchors  "
                  << "residual rms=" << std::fixed << std::setprecision(2) << w.resid_rms
                  << "  max=" << w.resid_max << "\n";
        std::cout.unsetf(std::ios::fixed);
        rbf_json.push_back({
            {"surface", n},
            {"n_anchors", w.n_anchors},
            {"resid_rms", w.resid_rms},
            {"resid_max", w.resid_max},
        });
    }

    std::cout << "[4.5] remapping UVs into reference frame\n";
    std::unordered_map<std::string, GMUV> uv_map;
    for (const auto& n : names) {
        const GMSurface& s = surfs.at(n);
        GMUV uv;
        warps[n].evalGrid(s.H, s.W, uv.uc, uv.ur);
        double cmin=1e18, cmax=-1e18, rmin=1e18, rmax=-1e18;
        for (int r = 0; r < s.H; ++r) {
            const uint8_t* mp = s.mask.ptr<uint8_t>(r);
            const float* uu = uv.uc.ptr<float>(r);
            const float* vv = uv.ur.ptr<float>(r);
            for (int c = 0; c < s.W; ++c) {
                if (!mp[c]) continue;
                if (!std::isfinite(uu[c]) || !std::isfinite(vv[c])) continue;
                if (uu[c] < cmin) cmin = uu[c];
                if (uu[c] > cmax) cmax = uu[c];
                if (vv[c] < rmin) rmin = vv[c];
                if (vv[c] > rmax) rmax = vv[c];
            }
        }
        std::cout << "  " << n << " UV in ref-frame: col ["
                  << std::fixed << std::setprecision(1) << cmin << ", " << cmax << "]"
                  << "  row [" << rmin << ", " << rmax << "]\n";
        std::cout.unsetf(std::ios::fixed);
        uv_map[n] = std::move(uv);
    }

    double all_cmin=1e18, all_cmax=-1e18, all_rmin=1e18, all_rmax=-1e18;
    for (const auto& n : names) {
        const GMSurface& s = surfs.at(n);
        const GMUV& uv = uv_map[n];
        for (int r = 0; r < s.H; ++r) {
            const uint8_t* mp = s.mask.ptr<uint8_t>(r);
            const float* uu = uv.uc.ptr<float>(r);
            const float* vv = uv.ur.ptr<float>(r);
            for (int c = 0; c < s.W; ++c) {
                if (!mp[c]) continue;
                if (uu[c] < all_cmin) all_cmin = uu[c];
                if (uu[c] > all_cmax) all_cmax = uu[c];
                if (vv[c] < all_rmin) all_rmin = vv[c];
                if (vv[c] > all_rmax) all_rmax = vv[c];
            }
        }
    }
    const int u_min = (int)std::floor(all_cmin);
    const int u_max = (int)std::ceil (all_cmax);
    const int v_min = (int)std::floor(all_rmin);
    const int v_max = (int)std::ceil (all_rmax);
    std::cout << "[5/6] N-way EDT blend  shared raster "
              << (v_max - v_min + 1) << "x" << (u_max - u_min + 1)
              << "  (U:[" << u_min << "," << u_max
              << "] V:[" << v_min << "," << v_max << "])\n";
    for (const auto& n : names) {
        const GMSurface& s = surfs.at(n);
        int ov = cv::countNonZero(real_overlap_native.at(n));
        std::cout << "  " << n << ": valid=" << s.valid
                  << "  real-overlap=" << ov
                  << "  private=" << (s.valid - ov) << "\n";
    }
    auto xyz_blended = gmEdtBlend(names, surfs, uv_map,
                                  u_min, u_max, v_min, v_max,
                                  real_overlap_native,
                                  kConsensusC, kConsensusMinActive,
                                  kIdwK);

    // Same pattern as stnRun: write OBJ to a sibling temp, rasterize (which
    // creates pre_dir), then move OBJ inside.
    const fs::path pre_obj_tmp = cfg.paths_dir / pre_obj.filename();
    std::cout << "[6/7] writing pre-straighten OBJ to " << pre_obj << "\n";
    {
        std::ofstream obj(pre_obj_tmp);
        if (!obj) {
            std::cerr << "cannot open OBJ for writing: " << pre_obj_tmp << "\n";
            return 1;
        }
        obj << "# Global merge: " << names.size() << " surfaces, ref=" << ref << "\n";
        int v_off=0, vt_off=0, total_v=0, total_f=0;
        for (const auto& n : names) {
            const GMSurface& s = surfs.at(n);
            const auto& bo = xyz_blended.at(n);
            const GMUV& uv = uv_map[n];
            auto [nv, nf] = gmEmitMesh(bo.X, bo.Y, bo.Z, s.mask, uv.uc, uv.ur,
                                       obj, v_off, vt_off, kStepSize);
            std::cout << "  " << n << ": " << nv << " verts, " << nf << " tris\n";
            v_off += nv; vt_off += nv;
            total_v += nv; total_f += nf;
        }
        std::cout << "  total: " << total_v << " verts, " << total_f << " tris\n";
    }

    std::cout << "rasterizing pre-straighten OBJ -> " << pre_dir
              << " (step=" << kStepSize << ")\n";
    if (const int rc = rasterizeObjToTifxyz(obj2tifxyz, pre_obj_tmp, pre_dir,
                                            kStepSize); rc != 0)
        return rc;
    fs::rename(pre_obj_tmp, pre_obj);
    reportTifxyzShape(pre_dir, "pre-straighten output");

    std::cout << "[7/7] straighten (alpha_u=" << cfg.straighten_strength_u
              << ", alpha_v=" << cfg.straighten_strength_v << ")\n";
    const int rrc = stnRun(pre_dir, output_dir, obj_out, obj2tifxyz,
                           kStepSize,
                           cfg.straighten_strength_u,
                           cfg.straighten_strength_v);
    if (rrc != 0) return rrc;

    json summary = {
        {"merge_json", merge_path.string()},
        {"output", output_dir.string()},
        {"obj_out", obj_out.string()},
        {"ref_surface", ref},
        {"surfaces", json::array()},
        {"edges", edges_json},
        {"joint_affine", joint_json},
        {"surface_rbf", rbf_json},
        {"params", {
            {"ridge_lambda", kRidgeLambda},
            {"consensus_c", kConsensusC},
            {"consensus_min_active", kConsensusMinActive},
            {"ransac_iters", cfg.ransac_iters},
            {"ransac_min_thresh", cfg.ransac_min_thresh},
            {"ransac_max_thresh", cfg.ransac_max_thresh},
            {"ransac_mad_k", cfg.ransac_mad_k},
            {"ransac_seed", cfg.ransac_seed},
            {"idw_k", kIdwK},
            {"step_size", kStepSize},
            {"straighten_strength_u", cfg.straighten_strength_u},
            {"straighten_strength_v", cfg.straighten_strength_v},
            {"anchor_bin_size", kAnchorBinSize},
        }},
    };
    for (const auto& n : names) {
        const GMSurface& s = surfs.at(n);
        summary["surfaces"].push_back({
            {"name", n}, {"path", s.path.string()},
            {"H", s.H}, {"W", s.W}, {"valid", s.valid},
        });
    }
    {
        std::ofstream f(summary_path);
        f << summary.dump(2) << std::endl;
        std::cout << "wrote summary to " << summary_path << "\n";
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    GMConfig defaults;
    po::options_description desc(
        "vc_merge_tifxyz: N-surface global tifxyz merge. The only required "
        "input is --merge <path/to/merge.json>; the surface list and edge "
        "graph come from that file (row-major grid of full tifxyz directory "
        "names), and <volpkg>/paths/ is the sibling directory holding the "
        "input tifxyz dirs. Output dirs are auto-named under <volpkg>/paths/: "
        "<alpha_first>_merged for the final tifxyz and "
        "<alpha_first>_merged_prestraightened for the intermediate, where "
        "<alpha_first> is the alphabetically smallest surface name in the "
        "grid. Runs per-edge overlap+anchor passes in-memory, RANSAC "
        "similarity, joint affine, per-surface TPS RBF, EDT blend, OBJ emit, "
        "vc_obj2tifxyz_legacy rasterization, and a final straighten pass.");
    desc.add_options()
        ("help,h", "print help")
        ("merge,m", po::value<std::string>(),
         "Path to <volpkg>/merge.json (required). The volpkg dir is its "
         "parent and paths_dir is <volpkg>/paths.")
        ("obj2tifxyz", po::value<std::string>()->default_value(""),
         "Path to vc_obj2tifxyz_legacy. Default: sibling binary next to "
         "vc_merge_tifxyz, falling back to PATH lookup.")
        ("ref", po::value<std::string>()->default_value(""),
         "Reference surface name. Empty (default) = pick the surface with "
         "the largest valid-cell count.")
        ("strength-u", po::value<double>()->default_value(defaults.straighten_strength_u),
         "Straighten theta-axis UV blend factor in [0, 1].")
        ("strength-v", po::value<double>()->default_value(defaults.straighten_strength_v),
         "Straighten z-axis UV blend factor in [0, 1].")
        ("ransac-iters", po::value<int>()->default_value(defaults.ransac_iters),
         "Per-edge RANSAC trial count.")
        ("ransac-min-thresh", po::value<double>()->default_value(defaults.ransac_min_thresh),
         "Per-edge RANSAC inlier-distance lower bound (vox).")
        ("ransac-max-thresh", po::value<double>()->default_value(defaults.ransac_max_thresh),
         "Per-edge RANSAC inlier-distance upper bound (vox).")
        ("ransac-mad-k", po::value<double>()->default_value(defaults.ransac_mad_k),
         "Per-edge RANSAC MAD multiplier (clamped to [min, max] threshold).")
        ("ransac-seed", po::value<uint32_t>()->default_value(defaults.ransac_seed),
         "Per-edge RANSAC RNG seed (0 = nondeterministic).");

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "error parsing arguments: " << e.what() << "\n" << desc << std::endl;
        return EXIT_FAILURE;
    }

    if (vm.count("help") || !vm.count("merge")) {
        std::cout << desc << std::endl;
        return vm.count("help") ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    const fs::path merge_path  = vm["merge"].as<std::string>();
    const fs::path obj2tifxyz  = resolveObj2Tifxyz(
        vm["obj2tifxyz"].as<std::string>(), argv[0]);

    GMConfig cfg;
    cfg.ref                   = vm["ref"].as<std::string>();
    cfg.straighten_strength_u = vm["strength-u"].as<double>();
    cfg.straighten_strength_v = vm["strength-v"].as<double>();
    cfg.ransac_iters          = vm["ransac-iters"].as<int>();
    cfg.ransac_min_thresh     = vm["ransac-min-thresh"].as<double>();
    cfg.ransac_max_thresh     = vm["ransac-max-thresh"].as<double>();
    cfg.ransac_mad_k          = vm["ransac-mad-k"].as<double>();
    cfg.ransac_seed           = vm["ransac-seed"].as<uint32_t>();

    try {
        return gmRunGlobalMerge(merge_path, obj2tifxyz, std::move(cfg));
    } catch (const std::exception& e) {
        std::cerr << "global merge failed: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}

