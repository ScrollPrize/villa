// vc_straighten.cpp
//
// Unified tifxyz straightening pipeline. Ports four Python prototypes
// (straighten_unbend_band, straighten_overlap_pairs,
// straighten_orthogonalize_columns, trim_stretch_outliers) into one binary
// with shared load / resample / save infrastructure.
//
// Stages, in pipeline order:
//   unbend        gentle global bend removal: fit a smooth spine through the
//                 band and re-coordinate into (arc-length, normal-offset). A
//                 local rotation, so it does not shear; run before the others
//                 when the band curves in UV.
//   overlap-pairs cross-wrap row alignment: pair each grid point with the
//                 nearest 3D point one winding over (its self-overlap), then
//                 warp v so paired points share a row. May be repeated.
//   orthogonalize remove residual column shear measured from the surface
//                 tangents (tau = e_v . e_u / |e_u|^2) via a u-warp.
//   trim          invalidate "bridge" cells whose 3D step to a grid neighbor
//                 is absurd (they wreck area-weighted distortion metrics).
//
// Default pipeline (no stage flags): unbend, overlap-pairs x2, orthogonalize,
// trim. Each warp uses a monotone (fold-over-proof) inverse and validity-aware
// bilinear resampling, matching the Python reference.

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Tiff.hpp"
#include "utils/Json.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/flann.hpp>

#include <boost/program_options.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace po = boost::program_options;
namespace fs = std::filesystem;
using utils::Json;

// ---------------------------------------------------------------------------
// Grid: a tifxyz surface in memory.
// ---------------------------------------------------------------------------
struct Grid {
    cv::Mat_<cv::Vec3f>  P;       // H x W world points; invalid = (-1,-1,-1)
    cv::Mat_<uint8_t>    valid;   // H x W; 1 = valid
    cv::Vec2f            scale{0.05f, 0.05f};
    int H() const { return P.rows; }
    int W() const { return P.cols; }
    long nvalid() const { return cv::countNonZero(valid); }
};

static bool isInvalid(const cv::Vec3f& p) {
    return p[0] == -1.0f && p[1] == -1.0f && p[2] == -1.0f;
}

static Grid loadGrid(const fs::path& dir)
{
    auto surf = load_quad_from_tifxyz(dir.string());
    Grid g;
    g.P = surf->rawPointsPtr()->clone();
    g.scale = surf->scale();
    const int H = g.P.rows, W = g.P.cols;
    g.valid = cv::Mat_<uint8_t>::zeros(H, W);
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            if (!isInvalid(g.P(r, c))) g.valid(r, c) = 1;

    // Honor an explicit mask.tif the same way the Python loader does.
    const fs::path mp = dir / "mask.tif";
    if (fs::exists(mp)) {
        std::vector<cv::Mat> layers;
        cv::imreadmulti(mp.string(), layers, cv::IMREAD_UNCHANGED);
        if (!layers.empty() && !layers[0].empty()
            && layers[0].rows == H && layers[0].cols == W) {
            cv::Mat m = layers[0];
            if (m.channels() > 1) { std::vector<cv::Mat> ch; cv::split(m, ch); m = ch[0]; }
            cv::Mat m8; m.convertTo(m8, CV_8U);
            for (int r = 0; r < H; ++r)
                for (int c = 0; c < W; ++c)
                    if (m8.at<uint8_t>(r, c) < 255) g.valid(r, c) = 0;
        }
    }
    return g;
}

static void saveGrid(const Grid& g, const fs::path& outdir, const fs::path& srcdir)
{
    if (fs::exists(outdir))
        throw std::runtime_error("output exists: " + outdir.string());
    fs::create_directories(outdir);

    cv::Mat_<cv::Vec3f> P = g.P.clone();
    for (int r = 0; r < g.H(); ++r)
        for (int c = 0; c < g.W(); ++c)
            if (!g.valid(r, c)) P(r, c) = cv::Vec3f(-1, -1, -1);

    std::vector<cv::Mat> xyz;
    cv::split(P, xyz);
    writeTiff(outdir / "x.tif", xyz[0], -1, 0, 0, -1.0f, COMPRESSION_ADOBE_DEFLATE);
    writeTiff(outdir / "y.tif", xyz[1], -1, 0, 0, -1.0f, COMPRESSION_ADOBE_DEFLATE);
    writeTiff(outdir / "z.tif", xyz[2], -1, 0, 0, -1.0f, COMPRESSION_ADOBE_DEFLATE);
    cv::Mat mask(g.H(), g.W(), CV_8U);
    for (int r = 0; r < g.H(); ++r)
        for (int c = 0; c < g.W(); ++c)
            mask.at<uint8_t>(r, c) = g.valid(r, c) ? 255 : 0;
    writeTiff(outdir / "mask.tif", mask, -1, 0, 0, 0.0f, COMPRESSION_ADOBE_DEFLATE);

    Json meta;
    const fs::path sm = srcdir / "meta.json";
    if (fs::exists(sm)) meta = Json::parse_file(sm.string());
    meta["uuid"] = outdir.filename().string();
    meta["format"] = "tifxyz";
    meta["type"] = "seg";
    if (!meta.contains("scale")) {           // preserve the input's scale array
        Json arr = Json::array();            // (resampling does not change it)
        Json a; a = (double)g.scale[0]; arr.push_back(a);
        Json b; b = (double)g.scale[1]; arr.push_back(b);
        meta["scale"] = arr;
    }
    meta.erase("bbox");
    std::ofstream out(outdir / "meta.json");
    out << meta.dump();
}

// ---------------------------------------------------------------------------
// Small numeric helpers (numpy/scipy equivalents).
// ---------------------------------------------------------------------------

// np.interp: piecewise-linear with endpoint clamping; xp strictly ascending.
static double interp1(const std::vector<double>& xp, const std::vector<double>& fp,
                      double x)
{
    const size_t n = xp.size();
    if (n == 0) return 0.0;
    if (x <= xp[0]) return fp[0];
    if (x >= xp[n - 1]) return fp[n - 1];
    size_t lo = 0, hi = n - 1;
    while (hi - lo > 1) { size_t m = (lo + hi) / 2; if (xp[m] <= x) lo = m; else hi = m; }
    const double t = (x - xp[lo]) / (xp[hi] - xp[lo]);
    return fp[lo] + t * (fp[hi] - fp[lo]);
}

// percentile p in [0,100] over a copy of v (numpy 'linear' interpolation).
static double percentile(std::vector<double> v, double p)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double idx = (p / 100.0) * (v.size() - 1);
    const size_t lo = (size_t)std::floor(idx);
    const size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (idx - lo) * (v[hi] - v[lo]);
}

// 1-D Gaussian smoothing, BORDER_REPLICATE, truncate=4 (matches scipy
// gaussian_filter1d mode='nearest').
static std::vector<double> gaussian1d(const std::vector<double>& a, double sigma)
{
    const int n = (int)a.size();
    if (sigma <= 0 || n == 0) return a;
    const int rad = (int)std::ceil(4.0 * sigma);
    std::vector<double> k(2 * rad + 1);
    double ksum = 0.0;
    for (int i = -rad; i <= rad; ++i) {
        k[i + rad] = std::exp(-0.5 * (i * i) / (sigma * sigma));
        ksum += k[i + rad];
    }
    for (double& w : k) w /= ksum;
    std::vector<double> out(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = -rad; j <= rad; ++j) {
            int idx = i + j;
            if (idx < 0) idx = 0; else if (idx >= n) idx = n - 1;  // replicate
            s += k[j + rad] * a[idx];
        }
        out[i] = s;
    }
    return out;
}

// np.gradient(A, h, axis): central differences interior, one-sided at edges.
static cv::Mat_<double> gradientAxis(const cv::Mat_<double>& A, double h, int axis)
{
    cv::Mat_<double> G(A.rows, A.cols, 0.0);
    if (axis == 0) {
        for (int c = 0; c < A.cols; ++c)
            for (int r = 0; r < A.rows; ++r) {
                if (A.rows == 1) { G(r, c) = 0; continue; }
                if (r == 0)            G(r, c) = (A(1, c) - A(0, c)) / h;
                else if (r == A.rows-1)G(r, c) = (A(r, c) - A(r-1, c)) / h;
                else                   G(r, c) = (A(r+1, c) - A(r-1, c)) / (2*h);
            }
    } else {
        for (int r = 0; r < A.rows; ++r)
            for (int c = 0; c < A.cols; ++c) {
                if (A.cols == 1) { G(r, c) = 0; continue; }
                if (c == 0)            G(r, c) = (A(r, 1) - A(r, 0)) / h;
                else if (c == A.cols-1)G(r, c) = (A(r, c) - A(r, c-1)) / h;
                else                   G(r, c) = (A(r, c+1) - A(r, c-1)) / (2*h);
            }
    }
    return G;
}

// Bilinear sampler over a coarse field at bin centers ((i+0.5)*bin), with
// linear extrapolation past the edge centers (matches RegularGridInterpolator
// method='linear', fill_value=None).
struct CoarseField {
    cv::Mat_<double> S;
    double binv, binu;
    double sample(double v, double u) const {
        const int nbv = S.rows, nbu = S.cols;
        double fi = v / binv - 0.5, fj = u / binu - 0.5;
        int i0 = (nbv >= 2) ? std::clamp((int)std::floor(fi), 0, nbv - 2) : 0;
        int j0 = (nbu >= 2) ? std::clamp((int)std::floor(fj), 0, nbu - 2) : 0;
        double ti = (nbv >= 2) ? (fi - i0) : 0.0;
        double tj = (nbu >= 2) ? (fj - j0) : 0.0;
        int i1 = std::min(i0 + 1, nbv - 1), j1 = std::min(j0 + 1, nbu - 1);
        double a = S(i0, j0) * (1 - tj) + S(i0, j1) * tj;
        double b = S(i1, j0) * (1 - tj) + S(i1, j1) * tj;
        return a * (1 - ti) + b * ti;
    }
};

// ---------------------------------------------------------------------------
// Validity-aware resampling + the shared shift-field machinery.
//
// Both warp stages reduce to the same operation, one the transpose of the
// other: bin a per-cell field, fill+smooth it, integrate into a cumulative
// row shift S(v,u), gauge-fix it, then resample v per column. orthogonalize
// reaches it by transposing its grid and (negated) field in and out, so this
// code only ever has to handle the v-warp orientation.
// ---------------------------------------------------------------------------

// Per-column v-warp: each (r,c) maps to source row fwd(r,c). Builds the
// monotone (fold-over-proof) inverse per column and blends with the
// both/only0/only1 rule.
static Grid resampleRows(const Grid& g, const cv::Mat_<double>& fwd)
{
    const int H = g.H(), W = g.W();
    double fmin = 1e300, fmax = -1e300;
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            if (g.valid(r, c)) { fmin = std::min(fmin, fwd(r, c)); fmax = std::max(fmax, fwd(r, c)); }
    const double offset = -fmin;
    const int nOut = (int)std::ceil(fmax + offset) + 1;

    Grid o;
    o.scale = g.scale;
    o.P = cv::Mat_<cv::Vec3f>(nOut, W, cv::Vec3f(-1, -1, -1));
    o.valid = cv::Mat_<uint8_t>::zeros(nOut, W);

    long dropped = 0;
    for (int c = 0; c < W; ++c) {
        std::vector<double> fk, vk;
        double run = -1e300;
        for (int r = 0; r < H; ++r) {
            const double f = fwd(r, c);
            if (f >= run) {
                if (fk.empty() || f - fk.back() > 1e-6) { fk.push_back(f); vk.push_back(r); }
                run = f;
            }
        }
        dropped += H - (long)fk.size();
        if (fk.size() < 2) continue;
        for (int o2 = 0; o2 < nOut; ++o2) {
            const double tgt = o2 - offset;
            if (tgt < fk.front() || tgt > fk.back()) continue;
            const double sc = interp1(fk, vk, tgt);
            if (sc < 0 || sc > H - 1) continue;
            const int r0 = (int)std::floor(sc), r1 = std::min(r0 + 1, H - 1);
            const float t = (float)(sc - r0);
            const bool m0 = g.valid(r0, c), m1 = g.valid(r1, c);
            if (!m0 && !m1) continue;
            o.P(o2, c) = (m0 && m1) ? (1.0f - t) * g.P(r0, c) + t * g.P(r1, c)
                       : (m0 ? g.P(r0, c) : g.P(r1, c));
            o.valid(o2, c) = 1;
        }
    }
    std::cout << "  monotone inverse: dropped " << dropped << " fold-over samples ("
              << (100.0 * dropped / ((double)H * W)) << "%), output "
              << nOut << "x" << W << " (offset " << offset << ")\n";
    return o;
}

static Grid transposeGrid(const Grid& g)
{
    Grid t;
    t.scale = cv::Vec2f(g.scale[1], g.scale[0]);
    cv::transpose(g.P, t.P);
    cv::transpose(g.valid, t.valid);
    return t;
}

// Fill NaN bins (interp along u within each row, then column means for fully
// empty rows, then 0) and Gaussian-smooth.
static void fillAndSmoothCoarse(cv::Mat_<double>& rate, double smooth)
{
    const int nbv = rate.rows, nbu = rate.cols;
    for (int i = 0; i < nbv; ++i) {
        std::vector<double> gx, gy;
        for (int j = 0; j < nbu; ++j) if (std::isfinite(rate(i, j))) { gx.push_back(j); gy.push_back(rate(i, j)); }
        if (!gx.empty() && (int)gx.size() < nbu)
            for (int j = 0; j < nbu; ++j) if (!std::isfinite(rate(i, j))) rate(i, j) = interp1(gx, gy, j);
    }
    std::vector<double> colMean(nbu, 0.0); std::vector<int> colCnt(nbu, 0);
    for (int i = 0; i < nbv; ++i)
        for (int j = 0; j < nbu; ++j)
            if (std::isfinite(rate(i, j))) { colMean[j] += rate(i, j); colCnt[j]++; }
    for (int j = 0; j < nbu; ++j) colMean[j] = colCnt[j] ? colMean[j] / colCnt[j] : 0.0;
    for (int i = 0; i < nbv; ++i) {
        bool anyFinite = false;
        for (int j = 0; j < nbu; ++j) if (std::isfinite(rate(i, j))) { anyFinite = true; break; }
        for (int j = 0; j < nbu; ++j)
            if (!anyFinite) rate(i, j) = colMean[j];
            else if (!std::isfinite(rate(i, j))) rate(i, j) = 0.0;
    }
    cv::GaussianBlur(rate, rate, cv::Size(0, 0), smooth, smooth, cv::BORDER_REPLICATE);
}

// Coarse per-bin count of valid full-res cells (gauge weights).
static cv::Mat_<double> coarseValidCount(const Grid& g, int binU, int binV, int nbv, int nbu)
{
    cv::Mat_<double> w(nbv, nbu, 0.0);
    for (int r = 0; r < g.H(); ++r)
        for (int c = 0; c < g.W(); ++c)
            if (g.valid(r, c)) w(std::min(r / binV, nbv - 1), std::min(c / binU, nbu - 1)) += 1.0;
    return w;
}

// Fill+smooth a binned rate field, integrate along u into a cumulative shift
// S(v,u), and apply the stretch-neutral gauge (zero the valid-weighted mean
// dS/dv per row). Returns the gauge-fixed field, ready for sampling.
static CoarseField buildShiftField(const Grid& g, cv::Mat_<double> rate,
                                   int binU, int binV, double smooth)
{
    fillAndSmoothCoarse(rate, smooth);
    const int nbv = rate.rows, nbu = rate.cols;
    cv::Mat_<double> S(nbv, nbu, 0.0);
    for (int i = 0; i < nbv; ++i) {
        double acc = 0.0;
        for (int j = 0; j < nbu; ++j) { acc += rate(i, j) * binU; S(i, j) = acc - 0.5 * rate(i, j) * binU; }
    }
    cv::Mat_<double> wValid = coarseValidCount(g, binU, binV, nbv, nbu);
    cv::Mat_<double> dSdv = gradientAxis(S, binV, 0);
    std::vector<double> gprime(nbv, 0.0);
    std::vector<int> goodRows; std::vector<double> gv;
    for (int i = 0; i < nbv; ++i) {
        double num = 0, den = 0;
        for (int j = 0; j < nbu; ++j) { num += dSdv(i, j) * wValid(i, j); den += wValid(i, j); }
        if (den > 0) { gprime[i] = num / den; goodRows.push_back(i); gv.push_back(gprime[i]); }
    }
    std::vector<double> gxd(goodRows.begin(), goodRows.end());
    for (int i = 0; i < nbv; ++i) {
        bool isGood = false; for (int gr : goodRows) if (gr == i) { isGood = true; break; }
        if (!isGood) gprime[i] = interp1(gxd, gv, i);
    }
    gprime = gaussian1d(gprime, 1.5);
    double acc = 0.0;
    for (int i = 0; i < nbv; ++i) { acc += gprime[i] * binV; for (int j = 0; j < nbu; ++j) S(i, j) -= acc; }
    return CoarseField{S, (double)binV, (double)binU};
}

// Forward map v' = v - S(v,u) and resample.
static Grid applyShiftWarp(const Grid& g, const CoarseField& field)
{
    cv::Mat_<double> fwd(g.H(), g.W());
    for (int r = 0; r < g.H(); ++r)
        for (int c = 0; c < g.W(); ++c)
            fwd(r, c) = r - field.sample(r, c);
    return resampleRows(g, fwd);
}

// ---------------------------------------------------------------------------
// Stage: unbend.
// ---------------------------------------------------------------------------
static Grid stageUnbend(const Grid& g, double smoothCols, int minValidRows)
{
    const int H = g.H(), W = g.W();
    std::cout << "[unbend] input " << H << "x" << W << ", valid " << g.nvalid() << "\n";

    // Spine: per-column median valid row, interpolated over empty columns.
    std::vector<double> spine(W, std::nan("")), cols(W);
    std::iota(cols.begin(), cols.end(), 0.0);
    std::vector<int> good;
    for (int c = 0; c < W; ++c) {
        std::vector<int> rows;
        for (int r = 0; r < H; ++r) if (g.valid(r, c)) rows.push_back(r);
        if ((int)rows.size() >= minValidRows) {
            std::nth_element(rows.begin(), rows.begin() + rows.size() / 2, rows.end());
            spine[c] = rows[rows.size() / 2];
            good.push_back(c);
        }
    }
    if ((int)good.size() < W / 10)
        throw std::runtime_error("unbend: too few populated columns for a spine");
    std::vector<double> gx, gy;
    for (int c : good) { gx.push_back(c); gy.push_back(spine[c]); }
    for (int c = 0; c < W; ++c)
        if (std::isnan(spine[c])) spine[c] = interp1(gx, gy, c);
    spine = gaussian1d(spine, smoothCols);

    // Tangent / normal / arc length along the spine.
    std::vector<double> fp(W), ds(W), s(W), nx(W), ny(W);
    for (int c = 0; c < W; ++c) {
        if (W == 1) fp[c] = 0;
        else if (c == 0)     fp[c] = spine[1] - spine[0];
        else if (c == W - 1) fp[c] = spine[c] - spine[c - 1];
        else                 fp[c] = 0.5 * (spine[c + 1] - spine[c - 1]);
    }
    for (int c = 0; c < W; ++c) {
        ds[c] = std::sqrt(1.0 + fp[c] * fp[c]);
        nx[c] = -fp[c] / ds[c];
        ny[c] = 1.0 / ds[c];
    }
    s[0] = 0.0;
    for (int c = 1; c < W; ++c) s[c] = s[c - 1] + 0.5 * (ds[c] + ds[c - 1]);

    // t-range used by valid cells (small-angle normal offset estimate).
    std::vector<double> tEst;
    tEst.reserve(g.nvalid());
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            if (g.valid(r, c)) tEst.push_back((r - spine[c]) / ds[c]);
    const double tLo = std::floor(percentile(tEst, 0.1)) - 10;
    const double tHi = std::ceil(percentile(tEst, 99.9)) + 10;

    const int Wout = (int)std::ceil(s[W - 1]) + 1;
    const int Hout = (int)(tHi - tLo) + 1;
    std::cout << "  spine arc length " << s[W - 1] << " cols, output "
              << Hout << "x" << Wout << " (t in [" << tLo << ", " << tHi << "])\n";

    // Per output column: source u(s), spine point, spine normal.
    std::vector<double> uOfS(Wout), sp(Wout), nxs(Wout), nys(Wout);
    for (int o = 0; o < Wout; ++o) {
        uOfS[o] = interp1(s, cols, (double)o);
        sp[o]  = interp1(cols, spine, uOfS[o]);
        nxs[o] = interp1(cols, nx, uOfS[o]);
        nys[o] = interp1(cols, ny, uOfS[o]);
    }

    Grid out;
    out.scale = g.scale;
    out.P = cv::Mat_<cv::Vec3f>(Hout, Wout, cv::Vec3f(-1, -1, -1));
    out.valid = cv::Mat_<uint8_t>::zeros(Hout, Wout);
    for (int oy = 0; oy < Hout; ++oy) {
        const double tv = oy + tLo;
        for (int ox = 0; ox < Wout; ++ox) {
            const double srcC = uOfS[ox] + tv * nxs[ox];
            const double srcR = sp[ox]   + tv * nys[ox];
            if (srcR < 0 || srcR > H - 1 || srcC < 0 || srcC > W - 1) continue;
            const int r0 = (int)std::floor(srcR), c0 = (int)std::floor(srcC);
            const int r1 = std::min(r0 + 1, H - 1), c1 = std::min(c0 + 1, W - 1);
            if (!g.valid(r0, c0) || !g.valid(r0, c1) ||
                !g.valid(r1, c0) || !g.valid(r1, c1)) continue;  // fully-valid only
            const float tr = (float)(srcR - r0), tc = (float)(srcC - c0);
            const cv::Vec3f a = (1 - tc) * g.P(r0, c0) + tc * g.P(r0, c1);
            const cv::Vec3f b = (1 - tc) * g.P(r1, c0) + tc * g.P(r1, c1);
            out.P(oy, ox) = (1 - tr) * a + tr * b;
            out.valid(oy, ox) = 1;
        }
    }
    std::cout << "  valid " << out.nvalid() << "\n";
    return out;
}

// ---------------------------------------------------------------------------
// Stage: overlap-pairs.
// ---------------------------------------------------------------------------
struct OverlapParams {
    int    stride = 2;
    double threshold = 0.0;     // 0 = auto
    double thresholdFactor = 1.4;
    int    minDu = 150;
    int    maxDv = 300;
    double maxSearch = 120.0;
    int    k = 32;
    int    binU = 64, binV = 64;
    int    minCount = 3;
    double smoothBins = 1.5;
    double clipRhoPct = 1.0;
    unsigned seed = 7;
};

// Returns the predicted post-warp mean|dv|; if applyWarp, *out holds the result.
static double stageOverlapPairs(const Grid& g, const OverlapParams& pr,
                                bool applyWarp, Grid* out)
{
    const int H = g.H(), W = g.W();
    std::cout << "[overlap-pairs] input " << H << "x" << W
              << ", valid " << g.nvalid() << "\n";

    // Subsample valid cells into a point cloud.
    std::vector<float> feat;   // N x 3
    std::vector<double> us, vs;
    for (int r = 0; r < H; r += pr.stride)
        for (int c = 0; c < W; c += pr.stride)
            if (g.valid(r, c)) {
                const cv::Vec3f& p = g.P(r, c);
                feat.push_back(p[0]); feat.push_back(p[1]); feat.push_back(p[2]);
                us.push_back(c); vs.push_back(r);
            }
    const int N = (int)us.size();
    std::cout << "  subsampled points (stride " << pr.stride << "): " << N << "\n";
    if (N < 1000) throw std::runtime_error("overlap-pairs: too few valid points");
    cv::Mat features(N, 3, CV_32F, feat.data());
    cv::flann::Index tree(features, cv::flann::KDTreeIndexParams(4));

    auto knn = [&](const cv::Mat& q, int k, cv::Mat& idx, cv::Mat& d2) {
        tree.knnSearch(q, idx, d2, k, cv::flann::SearchParams(32));
    };

    // Spacing estimate: nearest cross-wrap (|du|>=minDu) neighbor distance.
    // Skipped entirely when the user supplies --threshold, so an explicit
    // override is honored even on sparse surfaces where the random sample
    // would find too few cross-wrap neighbors to auto-calibrate.
    if (pr.threshold <= 0.0) {
        std::mt19937 rng(pr.seed);
        const int nSample = std::min(30000, N);
        std::vector<int> sel(N); std::iota(sel.begin(), sel.end(), 0);
        std::shuffle(sel.begin(), sel.end(), rng);
        sel.resize(nSample);
        cv::Mat q(nSample, 3, CV_32F);
        for (int i = 0; i < nSample; ++i)
            for (int j = 0; j < 3; ++j) q.at<float>(i, j) = feat[3 * sel[i] + j];
        cv::Mat idx, d2; knn(q, 64, idx, d2);
        std::vector<double> nd;
        const double maxS2 = pr.maxSearch * pr.maxSearch;
        for (int i = 0; i < nSample; ++i) {
            double best = 1e300;
            for (int kk = 0; kk < idx.cols; ++kk) {
                const int j = idx.at<int>(i, kk);
                if (j < 0 || j >= N) continue;
                const double dd2 = d2.at<float>(i, kk);
                if (dd2 > maxS2) continue;
                if (std::abs(us[j] - us[sel[i]]) >= pr.minDu)
                    best = std::min(best, std::sqrt(dd2));
            }
            if (best < 1e299) nd.push_back(best);
        }
        if (nd.size() < 100)
            throw std::runtime_error("overlap-pairs: spacing estimate found too few cross-wrap neighbors");
        const double med = percentile(nd, 50);
        std::cout << "  cross-wrap nearest distance p25/p50/p75 = "
                  << percentile(nd, 25) << "/" << med << "/" << percentile(nd, 75)
                  << " voxels (" << (100.0 * nd.size() / nSample) << "% matched)\n";
        const_cast<OverlapParams&>(pr).threshold = pr.thresholdFactor * med;
    }
    const double threshold = pr.threshold;
    const double thr2 = threshold * threshold;
    std::cout << "  threshold: " << threshold << " voxels\n";

    // Find one cross-wrap pair per point (nearest rightward neighbor).
    std::vector<int> pi, pj;
    {
        const int chunk = 200000;
        for (int lo = 0; lo < N; lo += chunk) {
            const int hi = std::min(lo + chunk, N);
            cv::Mat q(hi - lo, 3, CV_32F);
            for (int i = lo; i < hi; ++i)
                for (int j = 0; j < 3; ++j) q.at<float>(i - lo, j) = feat[3 * i + j];
            cv::Mat idx, d2; knn(q, pr.k, idx, d2);
            for (int i = lo; i < hi; ++i) {
                int bestj = -1; double bestd = 1e300;
                for (int kk = 0; kk < idx.cols; ++kk) {
                    const int j = idx.at<int>(i - lo, kk);
                    if (j < 0 || j >= N) continue;
                    const double dd2 = d2.at<float>(i - lo, kk);
                    if (dd2 > thr2) continue;
                    if (us[j] - us[i] < pr.minDu) continue;
                    if (std::abs(vs[j] - vs[i]) > pr.maxDv) continue;
                    if (dd2 < bestd) { bestd = dd2; bestj = j; }
                }
                if (bestj >= 0) { pi.push_back(i); pj.push_back(bestj); }
            }
        }
    }
    const int np = (int)pi.size();
    if (np < 1000) throw std::runtime_error("overlap-pairs: too few pairs");
    {
        std::vector<double> adv, advdu;
        double sdv = 0;
        for (int p = 0; p < np; ++p) {
            double dv = vs[pj[p]] - vs[pi[p]];
            adv.push_back(std::abs(dv)); sdv += dv;
            advdu.push_back(us[pj[p]] - us[pi[p]]);
        }
        std::cout << "  pairs: " << np << "  wrap width p50=" << percentile(advdu, 50)
                  << "  dv before: mean=" << sdv / np
                  << " mean|dv|=" << std::accumulate(adv.begin(), adv.end(), 0.0) / np
                  << " p95|dv|=" << percentile(adv, 95) << " rows\n";
    }

    // Rate field rho = dv/du binned at pair midpoints.
    const int nbu = (int)std::ceil((double)W / pr.binU);
    const int nbv = (int)std::ceil((double)H / pr.binV);
    std::vector<double> rho(np), um(np), vm(np);
    for (int p = 0; p < np; ++p) {
        rho[p] = (vs[pj[p]] - vs[pi[p]]) / (us[pj[p]] - us[pi[p]]);
        um[p] = 0.5 * (us[pi[p]] + us[pj[p]]);
        vm[p] = 0.5 * (vs[pi[p]] + vs[pj[p]]);
    }
    const double clo = percentile(rho, pr.clipRhoPct), chi = percentile(rho, 100 - pr.clipRhoPct);
    cv::Mat_<double> sum(nbv, nbu, 0.0), cnt(nbv, nbu, 0.0);
    for (int p = 0; p < np; ++p) {
        const double rc = std::clamp(rho[p], clo, chi);
        const int bu = std::clamp((int)(um[p] / pr.binU), 0, nbu - 1);
        const int bv = std::clamp((int)(vm[p] / pr.binV), 0, nbv - 1);
        sum(bv, bu) += rc; cnt(bv, bu) += 1.0;
    }
    cv::Mat_<double> rate(nbv, nbu); rate.setTo(std::nan(""));
    for (int i = 0; i < nbv; ++i)
        for (int j = 0; j < nbu; ++j)
            if (cnt(i, j) >= pr.minCount) rate(i, j) = sum(i, j) / cnt(i, j);

    CoarseField field = buildShiftField(g, rate, pr.binU, pr.binV, pr.smoothBins);

    // Predicted post-warp dv (validation).
    {
        double sdv = 0, sadv = 0; std::vector<double> adv;
        for (int p = 0; p < np; ++p) {
            const double s1 = field.sample(vs[pi[p]], us[pi[p]]);
            const double s2 = field.sample(vs[pj[p]], us[pj[p]]);
            const double d = (vs[pj[p]] - s2) - (vs[pi[p]] - s1);
            sdv += d; sadv += std::abs(d); adv.push_back(std::abs(d));
        }
        std::cout << "  dv after (predicted): mean=" << sdv / np
                  << " mean|dv|=" << sadv / np
                  << " p95|dv|=" << percentile(adv, 95) << " rows\n";
        if (!applyWarp) return sadv / np;
    }

    *out = applyShiftWarp(g, field);
    std::cout << "  valid " << out->nvalid() << "\n";
    return 0.0;
}

// ---------------------------------------------------------------------------
// Stage: orthogonalize columns.
// ---------------------------------------------------------------------------
struct OrthoParams {
    int binU = 64, binV = 64, minCount = 50;
    double smoothBins = 1.5, clipTauPct = 1.0;
};

// tau = (e_v . e_u)/|e_u|^2 from central-difference tangents; NaN where unavailable.
static cv::Mat_<double> measureSlant(const Grid& g)
{
    const int H = g.H(), W = g.W();
    cv::Mat_<double> tau(H, W); tau.setTo(std::nan(""));
    for (int r = 0; r < H; ++r)
        for (int c = 1; c < W - 1; ++c) {
            if (r < 1 || r > H - 2) continue;
            if (!g.valid(r, c + 1) || !g.valid(r, c - 1) ||
                !g.valid(r + 1, c) || !g.valid(r - 1, c)) continue;
            const cv::Vec3f eu = 0.5f * (g.P(r, c + 1) - g.P(r, c - 1));
            const cv::Vec3f ev = 0.5f * (g.P(r + 1, c) - g.P(r - 1, c));
            const double nu2 = eu.dot(eu);
            if (nu2 <= 0) continue;
            tau(r, c) = eu.dot(ev) / nu2;
        }
    return tau;
}

static Grid stageOrthogonalize(const Grid& g, const OrthoParams& pr)
{
    const int H = g.H(), W = g.W();
    std::cout << "[orthogonalize] input " << H << "x" << W << ", valid " << g.nvalid() << "\n";
    cv::Mat_<double> tau = measureSlant(g);
    std::vector<double> tv;
    for (int r = 0; r < H; ++r) for (int c = 0; c < W; ++c) if (std::isfinite(tau(r, c))) tv.push_back(tau(r, c));
    const double clo = percentile(tv, pr.clipTauPct), chi = percentile(tv, 100 - pr.clipTauPct);
    {
        std::vector<double> a; for (double v : tv) a.push_back(std::abs(std::clamp(v, clo, chi)));
        std::cout << "  slant before: mean|tau|=" << std::accumulate(a.begin(), a.end(), 0.0) / std::max<size_t>(a.size(),1)
                  << " p95=" << percentile(a, 95) << "\n";
    }

    const int nbu = (int)std::ceil((double)W / pr.binU);
    const int nbv = (int)std::ceil((double)H / pr.binV);
    cv::Mat_<double> sum(nbv, nbu, 0.0), cnt(nbv, nbu, 0.0);
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            if (std::isfinite(tau(r, c))) {
                sum(std::min(r / pr.binV, nbv - 1), std::min(c / pr.binU, nbu - 1)) += std::clamp(tau(r, c), clo, chi);
                cnt(std::min(r / pr.binV, nbv - 1), std::min(c / pr.binU, nbu - 1)) += 1.0;
            }
    cv::Mat_<double> fld(nbv, nbu); fld.setTo(std::nan(""));
    for (int i = 0; i < nbv; ++i) for (int j = 0; j < nbu; ++j) if (cnt(i, j) >= pr.minCount) fld(i, j) = sum(i, j) / cnt(i, j);

    // The column-shear u-warp is the transpose of the row-shift v-warp. Feed the
    // transposed grid and the transposed, negated slant field through the shared
    // shift machinery (negated so the shared v' = v - S yields u' = u + T), then
    // transpose the result back. Bin sizes swap with the axes.
    Grid gt = transposeGrid(g);
    cv::Mat_<double> rt; cv::transpose(fld, rt); rt = -rt;
    CoarseField field = buildShiftField(gt, rt, pr.binV, pr.binU, pr.smoothBins);
    Grid out = transposeGrid(applyShiftWarp(gt, field));
    out.scale = g.scale;

    cv::Mat_<double> tau2 = measureSlant(out);
    std::vector<double> a2;
    for (int r = 0; r < out.H(); ++r) for (int c = 0; c < out.W(); ++c) if (std::isfinite(tau2(r, c))) a2.push_back(std::abs(tau2(r, c)));
    std::cout << "  slant after: mean|tau|=" << std::accumulate(a2.begin(), a2.end(), 0.0) / std::max<size_t>(a2.size(),1)
              << " p95=" << percentile(a2, 95) << "\n  valid " << out.nvalid() << "\n";
    return out;
}

// ---------------------------------------------------------------------------
// Stage: trim bridge cells.
// ---------------------------------------------------------------------------
static void stageTrim(Grid& g, double maxEdge)
{
    const int H = g.H(), W = g.W();
    cv::Mat_<uint8_t> bad = cv::Mat_<uint8_t>::zeros(H, W);
    auto dist = [&](int r0, int c0, int r1, int c1) {
        return cv::norm(g.P(r0, c0) - g.P(r1, c1));
    };
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c) {
            if (!g.valid(r, c)) continue;
            if (c + 1 < W && g.valid(r, c + 1) && dist(r, c, r, c + 1) > maxEdge) { bad(r, c) = 1; bad(r, c + 1) = 1; }
            if (r + 1 < H && g.valid(r + 1, c) && dist(r, c, r + 1, c) > maxEdge) { bad(r, c) = 1; bad(r + 1, c) = 1; }
        }
    long n = 0;
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            if (bad(r, c) && g.valid(r, c)) { g.valid(r, c) = 0; g.P(r, c) = cv::Vec3f(-1, -1, -1); ++n; }
    std::cout << "[trim] removed " << n << " bridge cells (max-edge " << maxEdge
              << "); valid " << g.nvalid() << "\n";
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string inPath, outPath;
    OverlapParams op; OrthoParams qp;
    double unbendSmooth = 300.0; int unbendMinRows = 30;
    double trimMaxEdge = 100.0;
    int overlapCount = -1;
    bool doUnbend = false, doOrtho = false, doTrim = false, analyze = false;

    po::options_description desc("vc_straighten: unified tifxyz straightening");
    desc.add_options()
        ("help,h", "show help")
        ("input", po::value<std::string>(&inPath), "input tifxyz dir")
        ("output", po::value<std::string>(&outPath), "output tifxyz dir (must not exist)")
        ("unbend", po::bool_switch(&doUnbend), "run the spine unbend stage")
        ("overlap-pairs", po::value<int>(&overlapCount), "run N overlap-pair passes")
        ("orthogonalize", po::bool_switch(&doOrtho), "run the column orthogonalize stage")
        ("trim", po::bool_switch(&doTrim), "run the bridge-cell trim stage")
        ("analyze", po::bool_switch(&analyze), "measure cross-wrap dv on the input and exit (no output)")
        ("unbend-smooth-cols", po::value<double>(&unbendSmooth)->default_value(300.0), "unbend spine Gaussian sigma (cols)")
        ("unbend-min-rows", po::value<int>(&unbendMinRows)->default_value(30), "min valid rows for a spine column")
        ("threshold", po::value<double>(&op.threshold)->default_value(0.0), "overlap 3D threshold (0=auto)")
        ("min-du", po::value<int>(&op.minDu)->default_value(150), "min |du| (cols) for a cross-wrap pair")
        ("max-dv", po::value<int>(&op.maxDv)->default_value(300), "reject pairs with |dv| above this")
        ("stride", po::value<int>(&op.stride)->default_value(2), "subsample stride for the KD-tree")
        ("trim-max-edge", po::value<double>(&trimMaxEdge)->default_value(100.0), "trim cells touching a 3D edge longer than this (voxels)")
        ("seed", po::value<unsigned>(&op.seed)->default_value(7), "RNG seed for the spacing estimate");
    po::positional_options_description pos;
    pos.add("input", 1).add("output", 1);
    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(desc).positional(pos).run(), vm);
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n" << desc << "\n";
        return 1;
    }
    if (vm.count("help") || inPath.empty()) { std::cout << desc << "\n"; return vm.count("help") ? 0 : 1; }

    // Default pipeline when no stage flag is given.
    const bool anyStage = doUnbend || doOrtho || doTrim || overlapCount >= 0 || analyze;
    if (!anyStage) { doUnbend = true; overlapCount = 2; doOrtho = true; doTrim = true; }
    if (overlapCount < 0) overlapCount = 0;

    Grid g;
    try { g = loadGrid(inPath); }
    catch (const std::exception& e) { std::cerr << "error loading " << inPath << ": " << e.what() << "\n"; return 1; }

    if (analyze) {
        stageOverlapPairs(g, op, /*applyWarp=*/false, nullptr);
        return 0;
    }
    if (outPath.empty()) { std::cerr << "error: output dir required\n"; return 1; }
    if (fs::exists(outPath)) { std::cerr << "error: output exists: " << outPath << "\n"; return 1; }

    try {
        if (doUnbend) g = stageUnbend(g, unbendSmooth, unbendMinRows);
        for (int p = 0; p < overlapCount; ++p) {
            std::cout << "--- overlap-pairs pass " << (p + 1) << "/" << overlapCount << " ---\n";
            OverlapParams pp = op; pp.threshold = (op.threshold > 0 ? op.threshold : 0.0);
            Grid next; stageOverlapPairs(g, pp, /*applyWarp=*/true, &next); g = std::move(next);
        }
        if (doOrtho) g = stageOrthogonalize(g, qp);
        if (doTrim)  stageTrim(g, trimMaxEdge);
        saveGrid(g, outPath, inPath);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
    std::cout << "wrote: " << outPath << "  (" << g.H() << "x" << g.W()
              << ", valid=" << g.nvalid() << ")\n";
    return 0;
}
