// vc_coord_regression: Deterministic regression harness for the
// coord-math hot path (at_int / search_min_loc / pointTo).
//
// Covers three surface types so the three failure modes of a
// Newton-style pointTo replacement all surface here:
//   • synthetic smooth grid    — Gauss-Newton should converge cleanly
//   • synthetic curled/twisted — tests non-convex local minima
//   • loaded tifxyz segment    — real Vesuvius scroll geometry
//
// And three test kinds per surface:
//   AT  at_int(surface, (u,v))          — pure interp
//   PT  pointTo(init_loc, surface, tgt) — inverse search from seed
//   RT  round-trip: gen(u,v)→P, then pointTo(P) must return loc≈(u,v)
//       within surface-scale tolerance. Fails if the search diverges
//       to a different local minimum.
//
// Workflow:
//   $ vc_coord_regression > /tmp/baseline.txt
//   # (modify pointTo / search_min_loc / at_int)
//   $ vc_coord_regression > /tmp/after.txt
//   $ compare_coord_regression.py /tmp/baseline.txt /tmp/after.txt
//
// For round-trip cases, epsilon tolerance is looser on the loc output
// (several voxels) because pointTo finds a local minimum — tiny numeric
// differences can nudge to a nearby valley. The regression script
// classifies RT differently from AT/PT (accepts larger drift so long
// as the found 3D point P is close to the target).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>

#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/QuadSurface.hpp"

using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

namespace {

struct Surface {
    std::string label;
    cv::Mat_<cv::Vec3f> grid;
    cv::Vec2f scale;  // grid units per world unit (used to build QuadSurface)
};

Surface makeSmoothGrid(int w, int h) {
    Surface s;
    s.label = "smooth";
    s.grid.create(h, w);
    for (int y = 0; y < h; ++y) {
        auto* row = s.grid.ptr<cv::Vec3f>(y);
        for (int x = 0; x < w; ++x) {
            const float fx = float(x), fy = float(y);
            const float z = 100.0f + 5.0f * std::sin(fx * 0.013f)
                                   + 3.0f * std::cos(fy * 0.017f);
            row[x] = cv::Vec3f(fx * 1.10f, fy * 1.05f, z);
        }
    }
    s.scale = cv::Vec2f(1.0f, 1.0f);
    return s;
}

// Highly curved — tests convergence on non-monotonic surfaces. The
// sinusoidal warp creates ridges and valleys that can trap any
// gradient-descent search at a wrong local minimum.
Surface makeCurledGrid(int w, int h) {
    Surface s;
    s.label = "curled";
    s.grid.create(h, w);
    for (int y = 0; y < h; ++y) {
        auto* row = s.grid.ptr<cv::Vec3f>(y);
        for (int x = 0; x < w; ++x) {
            const float fx = float(x), fy = float(y);
            const float z = 100.0f + 30.0f * std::sin(fx * 0.05f)
                                   + 20.0f * std::sin(fy * 0.07f)
                                   + 15.0f * std::sin((fx + fy) * 0.09f);
            const float dx = 8.0f * std::sin(fy * 0.04f);
            const float dy = 6.0f * std::sin(fx * 0.035f);
            row[x] = cv::Vec3f(fx * 1.10f + dx, fy * 1.05f + dy, z);
        }
    }
    s.scale = cv::Vec2f(1.0f, 1.0f);
    return s;
}

// Loaded tifxyz segment. Uses load_quad_from_tifxyz so the resulting
// grid matches exactly what VC3D loads for a real segment.
std::optional<Surface> loadTifxyz(const std::string& path) {
    auto surf = load_quad_from_tifxyz(path, 0);
    if (!surf) return std::nullopt;
    Surface s;
    s.label = "tifxyz:" + fs::path(path).filename().string();
    s.grid = surf->rawPoints().clone();
    s.scale = surf->scale();
    return s;
}

struct CoordCase { int id; cv::Vec2f p; };
struct PtCase    { int id; cv::Vec3f target; cv::Vec2f initLoc; };
struct RtCase    { int id; cv::Vec2f trueLoc; };

std::vector<CoordCase> genAtIntCases(int w, int h, int n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ux(1.0f, float(w - 2));
    std::uniform_real_distribution<float> uy(1.0f, float(h - 2));
    std::vector<CoordCase> out; out.reserve(n);
    for (int i = 0; i < n; ++i) out.push_back({i, cv::Vec2f(ux(rng), uy(rng))});
    return out;
}

// pointTo cases: pick a random (u,v), sample the surface to get a real
// 3D point, then perturb it slightly. The init_loc is a different random
// (u,v) that isn't necessarily near the truth. This stresses the
// search's ability to converge from a distant seed.
std::vector<PtCase> genPtCases(const cv::Mat_<cv::Vec3f>& grid, int n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ux(5.0f, float(grid.cols) - 5.0f);
    std::uniform_real_distribution<float> uy(5.0f, float(grid.rows) - 5.0f);
    std::uniform_real_distribution<float> off(-1.5f, 1.5f);
    std::vector<PtCase> out; out.reserve(n);
    for (int i = 0; i < n; ++i) {
        const cv::Vec2f truePt(ux(rng), uy(rng));
        const cv::Vec3f tgt = at_int(grid, truePt)
            + cv::Vec3f(off(rng), off(rng), off(rng));
        const cv::Vec2f init(ux(rng), uy(rng));
        out.push_back({i, tgt, init});
    }
    return out;
}

// Round-trip cases: sample a random (u,v), record the 3D point, then
// we'll pointTo back to it with a seed nearby. Expect convergence to
// within a small tolerance of truePt.
std::vector<RtCase> genRtCases(const cv::Mat_<cv::Vec3f>& grid, int n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ux(10.0f, float(grid.cols) - 10.0f);
    std::uniform_real_distribution<float> uy(10.0f, float(grid.rows) - 10.0f);
    std::vector<RtCase> out; out.reserve(n);
    for (int i = 0; i < n; ++i) {
        out.push_back({i, cv::Vec2f(ux(rng), uy(rng))});
    }
    return out;
}

double now_sec(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

void runSurface(const Surface& s, int atN, int ptN, int rtN,
                bool noTiming)
{
    const std::string tagBase = s.label;
    const auto& grid = s.grid;
    printf("# surface: %s shape=%dx%d scale=%.4f,%.4f\n",
           s.label.c_str(), grid.cols, grid.rows, s.scale[0], s.scale[1]);

    // AT cases
    {
        auto cases = genAtIntCases(grid.cols, grid.rows, atN, 0xA7F13Du);
        std::vector<cv::Vec3f> results(cases.size());
        auto t0 = Clock::now();
        cv::Vec3f sink{0,0,0};
        for (size_t i = 0; i < cases.size(); ++i) {
            results[i] = at_int(grid, cases[i].p);
            sink += results[i];
        }
        double sec = now_sec(t0);
        for (size_t i = 0; i < cases.size(); ++i) {
            const auto& r = results[i];
            printf("AT %s/%d %.9f %.9f %.9f\n", tagBase.c_str(), cases[i].id,
                   r[0], r[1], r[2]);
        }
        if (!noTiming)
            fprintf(stderr, "[%s] at_int %d in %.4fs (%.2f ns/call)\n",
                    tagBase.c_str(), atN, sec, sec * 1e9 / std::max(1, atN));
        volatile float used = sink[0] + sink[1] + sink[2]; (void)used;
    }

    // PT cases
    {
        auto cases = genPtCases(grid, ptN, 0xD42EA9u);
        struct PtResult { cv::Vec2f loc; cv::Vec3f p; float d; };
        std::vector<PtResult> results(cases.size());
        auto t0 = Clock::now();
        for (size_t i = 0; i < cases.size(); ++i) {
            cv::Vec2f loc = cases[i].initLoc;
            float d = pointTo(loc, grid, cases[i].target, 0.5f, 100, 1.0f);
            cv::Vec3f p = (loc[0] > 0) ? at_int(grid, loc) : cv::Vec3f{-1,-1,-1};
            results[i] = {loc, p, d};
        }
        double sec = now_sec(t0);
        for (size_t i = 0; i < cases.size(); ++i) {
            const auto& r = results[i];
            printf("PT %s/%d %.9f %.9f %.9f %.9f %.9f %.9f\n",
                   tagBase.c_str(), cases[i].id,
                   r.loc[0], r.loc[1], r.p[0], r.p[1], r.p[2], r.d);
        }
        if (!noTiming)
            fprintf(stderr, "[%s] pointTo %d in %.4fs (%.3f us/call)\n",
                    tagBase.c_str(), ptN, sec, sec * 1e6 / std::max(1, ptN));
    }

    // Round-trip cases: gen a random truePt, compute P, pointTo(P) with
    // a seed adjacent to truePt. Record the found loc + 3D distance to
    // target. Numerically sensitive fields prefix the tolerant metric
    // (pErr) so the compare script can relax per-kind tolerance.
    {
        auto cases = genRtCases(grid, rtN, 0xF00B4Eu);
        struct RtResult { cv::Vec2f truLoc, foundLoc; cv::Vec3f target, foundP; float locDrift, pErr; };
        std::vector<RtResult> results(cases.size());
        auto t0 = Clock::now();
        for (size_t i = 0; i < cases.size(); ++i) {
            const cv::Vec2f truLoc = cases[i].trueLoc;
            const cv::Vec3f target = at_int(grid, truLoc);
            // Seed a bit off the truth so the search has to converge.
            cv::Vec2f seedLoc = truLoc + cv::Vec2f(2.5f, -1.7f);
            if (seedLoc[0] < 2) seedLoc[0] = 2;
            if (seedLoc[1] < 2) seedLoc[1] = 2;
            if (seedLoc[0] > float(grid.cols - 3)) seedLoc[0] = float(grid.cols - 3);
            if (seedLoc[1] > float(grid.rows - 3)) seedLoc[1] = float(grid.rows - 3);
            cv::Vec2f loc = seedLoc;
            pointTo(loc, grid, target, 0.1f, 200, 1.0f);
            cv::Vec3f foundP = (loc[0] > 0) ? at_int(grid, loc) : cv::Vec3f{-1,-1,-1};
            cv::Vec2f delta = loc - truLoc;
            float locDrift = std::sqrt(delta[0]*delta[0] + delta[1]*delta[1]);
            cv::Vec3f err = foundP - target;
            float pErr = std::sqrt(err[0]*err[0] + err[1]*err[1] + err[2]*err[2]);
            results[i] = {truLoc, loc, target, foundP, locDrift, pErr};
        }
        double sec = now_sec(t0);
        // Output: true_u true_v found_u found_v locDrift pErr. The
        // critical quality metric is pErr — how close the found point
        // is to the target. locDrift can legitimately differ between
        // algorithms if they find different valid loc answers.
        for (size_t i = 0; i < cases.size(); ++i) {
            const auto& r = results[i];
            printf("RT %s/%d %.4f %.4f %.4f %.4f %.6f %.6f\n",
                   tagBase.c_str(), cases[i].id,
                   r.truLoc[0], r.truLoc[1], r.foundLoc[0], r.foundLoc[1],
                   r.locDrift, r.pErr);
        }
        if (!noTiming) {
            // Also summarise convergence quality — we care about this
            // even when baseline and candidate diverge in locDrift:
            // the algorithmic question is "does it converge close
            // enough to the target in 3D space?"
            std::vector<float> errs; errs.reserve(results.size());
            for (auto& r : results) errs.push_back(r.pErr);
            std::sort(errs.begin(), errs.end());
            const float med = errs[errs.size()/2];
            const float p95 = errs[std::min<size_t>(errs.size()-1, size_t(errs.size()*0.95))];
            fprintf(stderr,
                "[%s] round-trip %d in %.4fs (%.3f us/call)  "
                "pErr med=%.4f p95=%.4f\n",
                tagBase.c_str(), rtN, sec, sec * 1e6 / std::max(1, rtN),
                med, p95);
        }
    }
}

}  // namespace

int main(int argc, char** argv)
{
    int gridSize = 256;
    int atN = 500;
    int ptN = 100;
    int rtN = 100;
    bool noTiming = false;
    std::vector<std::string> tifxyzPaths;
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        auto need = [&](const char* w) { if (i+1>=argc){fprintf(stderr,"%s needs value\n",w);std::exit(1);} return argv[++i]; };
        if      (a == "--grid")        gridSize = std::atoi(need("--grid"));
        else if (a == "--at-int")      atN      = std::atoi(need("--at-int"));
        else if (a == "--point-to")    ptN      = std::atoi(need("--point-to"));
        else if (a == "--round-trip")  rtN      = std::atoi(need("--round-trip"));
        else if (a == "--no-timing")   noTiming = true;
        else if (a == "--tifxyz")      tifxyzPaths.emplace_back(need("--tifxyz"));
        else { fprintf(stderr, "Unknown: %s\n", argv[i]); return 1; }
    }

    printf("# vc_coord_regression grid=%d at-int=%d point-to=%d round-trip=%d\n",
           gridSize, atN, ptN, rtN);

    runSurface(makeSmoothGrid(gridSize, gridSize), atN, ptN, rtN, noTiming);
    runSurface(makeCurledGrid(gridSize, gridSize), atN, ptN, rtN, noTiming);
    for (const auto& p : tifxyzPaths) {
        auto s = loadTifxyz(p);
        if (!s) {
            fprintf(stderr, "# failed to load tifxyz: %s\n", p.c_str());
            continue;
        }
        runSurface(*s, atN, ptN, rtN, noTiming);
    }
    return 0;
}
