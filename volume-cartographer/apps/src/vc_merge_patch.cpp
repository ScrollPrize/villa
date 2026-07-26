// vc_merge_patch
//
// One-parent, one-child tifxyz patch merge. The child tifxyz is aligned
// to the parent so its border overlaps the parent's surface; the child's
// border region is blended into the parent and the child's interior
// replaces the corresponding region of the parent in-place. Parent cells
// outside the child footprint are written verbatim from the input parent
// (no rasterization, no UV transform).
//
// Pipeline:
//   [0/6] load parent + child tifxyz as QuadSurfaces
//   [1/6] derive child border mask from a distance transform of child mask
//   [2/6] build SurfacePatchIndex over both
//   [3/6] threshold sweep on child<->parent overlap, intersected with the
//         child border so anchors only seed the rim
//   [4/6] RANSAC similarity child->parent + TPS RBF on inlier residuals
//         (parent pinned to identity; only the child gets a warp)
//   [5/6] rasterize child footprint into parent grid; for each parent cell
//         in the footprint, KDTree-IDW sample child XYZ. Blend by the
//         child's distance-to-boundary: weight ramps 0 -> 1 over the
//         outermost `blend_cells` of the child. Inside that thin rim the
//         child fully replaces the parent.
//   [6/6] parent->saveOverwrite() + writeValidMask. The save snapshots
//         the last-persisted state into <parent_dir>/backups/<parent>/{0..7}/
//         (rotating 8-slot ring, same convention as grow / brush /
//         rotation edits) and then atomically swaps the new
//         x/y/z/mask/meta into place. Cells outside the footprint are
//         written verbatim; aux files in the parent dir (approval.tif,
//         generations.tif, etc.) are preserved.
//
// Hard-coded fitting params (kThresholds, kRidgeLambda, kAnchorBinSize)
// match vc_merge_tifxyz; the per-run CLI knobs cover RANSAC, border
// width, and the per-surface anchor cap that protects the dense TPS LU.

#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/OpenCvCompat.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/flann.hpp>

#include <Eigen/Dense>

#include <boost/program_options.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace po = boost::program_options;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

// =============================================================================
// Constants copied verbatim from vc_merge_tifxyz.cpp. The threshold sweep and
// per-cell density / normal-agreement filter have been tuned across volpkgs and
// are intentionally kept in lockstep with the global merger so anchor quality
// stays comparable between the two tools.
// =============================================================================
constexpr float  kThresholds[]      = {4.0f, 5.0f, 6.0f, 7.0f};
constexpr float  kNormalThresh      = 0.85f;
constexpr int    kDensityRadius     = 7;
constexpr float  kDensityThresh     = 0.5f;
constexpr int    kMinComponent      = 200;
constexpr int    kAnchorBinSize     = 1;
constexpr int    kDefaultBorderCells = 16;
constexpr int    kDefaultBlendCells  = 6;

using Mat1f = cv::Mat_<float>;
using Mat1b = cv::Mat_<uint8_t>;

// =============================================================================
// Overlap detection (copied from vc_merge_tifxyz.cpp).
// =============================================================================

struct OverlapMaps {
    Mat1f distance;    // -1 where no match, else euclidean distance to nearest B
    Mat1f normAgree;   // NaN where no match, else |nA . nB|
    Mat1b mask;        // 0/1 raw match mask
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
    m.distance  = Mat1f(H, W, -1.0f);
    m.normAgree = Mat1f(H, W, std::numeric_limits<float>::quiet_NaN());
    m.mask      = Mat1b(H, W, (uint8_t)0);

    (void)a->gridNormal(0, 0);
    (void)b->rawPoints();

    #pragma omp parallel for schedule(dynamic, 16)
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            const cv::Vec3f& p = pts(r, c);
            if (p[0] < 0) continue;

            SurfacePatchIndex::PointQuery query;
            query.worldPoint = p;
            query.tolerance = threshold;
            query.surfaces.only = b;
            auto res = idx.locate(query);
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

Mat1f boxDensity(const Mat1b& mask, int radius)
{
    Mat1f src;
    mask.convertTo(src, CV_32F);
    Mat1f dst;
    const int k = 2 * radius + 1;
    cv::boxFilter(src, dst, CV_32F, cv::Size(k, k), cv::Point(-1, -1),
                  /*normalize=*/true, cv::BORDER_REFLECT);
    return dst;
}

int largestComponent(Mat1b& mask, int minSize)
{
    cv::Mat labels, stats, cent;
    const int n = cv::connectedComponentsWithStats(mask, labels, stats, cent, 8, CV_32S);
    int kept = 0;
    std::vector<uint8_t> keep(n, 0);
    for (int i = 1; i < n; ++i) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= minSize) {
            keep[i] = 1; ++kept;
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
    long   matches{0};
    long   valid{0};
    double distMedian{std::numeric_limits<double>::quiet_NaN()};
    double distMean{std::numeric_limits<double>::quiet_NaN()};
    double normMedian{std::numeric_limits<double>::quiet_NaN()};
    double normMean{std::numeric_limits<double>::quiet_NaN()};
    long   realCells{0};
    long   realComponents{0};
};

Mat1b maskAtThreshold(const Mat1f& dist, float t)
{
    Mat1b m(dist.size(), (uint8_t)0);
    for (int y = 0; y < dist.rows; ++y) {
        for (int x = 0; x < dist.cols; ++x) {
            const float d = dist(y, x);
            if (d >= 0.0f && d < t) m(y, x) = 1;
        }
    }
    return m;
}

Stats summarize(const cv::Mat_<cv::Vec3f>& pts,
                const Mat1f& distMap, const Mat1f& normMap,
                const Mat1b& matchMask, Mat1b& realMask)
{
    Stats s;
    std::vector<double> dists, norms;
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
        double sum = 0;
        for (double x : v) sum += x;
        return sum / v.size();
    };
    s.distMean   = mean(dists);
    s.distMedian = med(dists);
    s.normMean   = mean(norms);
    s.normMedian = med(norms);

    Mat1f density = boxDensity(matchMask, kDensityRadius);
    realMask = Mat1b(matchMask.size(), (uint8_t)0);
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

// =============================================================================
// Anchor selection (copied from vc_merge_tifxyz.cpp). Caller is expected to
// hand us a realMaskA already intersected with the child border so anchors
// only seed the rim.
// =============================================================================

struct Anchor {
    float     a_row{0}, a_col{0};
    cv::Vec3f a_world{0, 0, 0};
    float     b_row{0}, b_col{0};
    cv::Vec3f b_world{0, 0, 0};
    float     distance{0};
    float     normal_agree{std::numeric_limits<float>::quiet_NaN()};
};

std::vector<Anchor> pickAnchors(const std::shared_ptr<QuadSurface>& A,
                                const std::shared_ptr<QuadSurface>& B,
                                const SurfacePatchIndex& idx,
                                const Mat1b& realMaskA,
                                const Mat1f& distMapA,
                                const Mat1f& normMapA,
                                int binSize, float locateTolerance)
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
            int bestR = -1, bestC = -1;
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
            SurfacePatchIndex::PointQuery query;
            query.worldPoint = aw;
            query.tolerance = locateTolerance;
            query.surfaces.only = B;
            auto res = idx.locate(query);
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
// Affine + RANSAC + TPS RBF helpers (copied from vc_merge_tifxyz.cpp).
// vc_merge_patch is a single-edge case (N=2 with parent pinned), so the
// joint-affine sparse-QR machinery from the global merger is unnecessary --
// the RANSAC similarity is used directly as the linear part of the warp and
// the TPS RBF handles any residual non-linear distortion.
// =============================================================================

inline cv::Vec2d pmApplyAffine(const cv::Matx23d& M, const cv::Vec2d& p)
{
    return cv::Vec2d(M(0,0)*p[0] + M(0,1)*p[1] + M(0,2),
                     M(1,0)*p[0] + M(1,1)*p[1] + M(1,2));
}

cv::Matx23d pmFitSimilarity(const std::vector<cv::Vec2d>& src,
                            const std::vector<cv::Vec2d>& dst)
{
    const int N = (int)src.size();
    cv::Matx23d M = cv::Matx23d::eye();
    if (N == 0) return M;
    cv::Vec2d mu_s(0, 0), mu_d(0, 0);
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
    if (U.determinant() * V.determinant() < 0) D(1, 1) = -1.0;
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

inline double pmSimilarityScale(const cv::Matx23d& M)
{
    return std::sqrt(std::abs(M(0,0)*M(1,1) - M(0,1)*M(1,0)));
}

double pmMedianInPlace(std::vector<double>& v)
{
    if (v.empty()) return 0.0;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

struct PMRansac {
    cv::Matx23d M;
    std::vector<uint8_t> inlier;
    double thresh{0.0};
    double sigma_in{1.0};
    int    n_in{0};
};

PMRansac pmRansacSimilarity(const std::vector<cv::Vec2d>& src,
                            const std::vector<cv::Vec2d>& dst,
                            int iters, double min_t, double max_t,
                            double mad_k, uint32_t seed)
{
    const int N = (int)src.size();
    PMRansac R; R.inlier.assign(N, 1);
    if (N < 2) {
        R.M = cv::Matx23d::eye();
        R.thresh = min_t; R.n_in = N;
        return R;
    }
    cv::Matx23d M0 = pmFitSimilarity(src, dst);
    std::vector<double> r0(N);
    for (int i = 0; i < N; ++i) r0[i] = cv::norm(dst[i] - pmApplyAffine(M0, src[i]));
    auto r0sorted = r0;
    double med = pmMedianInPlace(r0sorted);
    std::vector<double> dev(N);
    for (int i = 0; i < N; ++i) dev[i] = std::abs(r0[i] - med);
    double mad = pmMedianInPlace(dev) * 1.4826;
    double thresh = std::min(max_t, std::max(min_t, mad_k * mad));

    // seed==0 means "pick nondeterministically", matching the CLI doc.
    std::mt19937 rng(seed ? seed : std::random_device{}());
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
        try { Mh = pmFitSimilarity(s2, d2); } catch (...) { continue; }
        int cnt = 0;
        std::vector<uint8_t> inl(N, 0);
        for (int k = 0; k < N; ++k) {
            const double r = cv::norm(dst[k] - pmApplyAffine(Mh, src[k]));
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
    cv::Matx23d M = pmFitSimilarity(s_inl, d_inl);
    for (int k = 0; k < N; ++k) {
        const double r = cv::norm(dst[k] - pmApplyAffine(M, src[k]));
        best_inl[k] = (r < thresh) ? 1 : 0;
    }
    gather(s_inl, d_inl);
    M = pmFitSimilarity(s_inl, d_inl);
    double sse = 0.0; int Nin = 0;
    for (int k = 0; k < N; ++k) {
        if (!best_inl[k]) continue;
        const double r = cv::norm(dst[k] - pmApplyAffine(M, src[k]));
        sse += r * r; ++Nin;
    }
    R.M = M;
    R.inlier = std::move(best_inl);
    R.thresh = thresh;
    R.sigma_in = (Nin > 0) ? std::sqrt(sse / Nin) : 1.0;
    R.n_in = Nin;
    return R;
}

inline double pmTpsKernel(double r2)
{
    if (r2 <= 0.0) return 0.0;
    return r2 * 0.5 * std::log(r2);
}

struct PMSimRBF {
    cv::Matx23d M{cv::Matx23d::eye()};
    Eigen::MatrixXd anchors;
    Eigen::MatrixXd weights;
    Eigen::MatrixXd poly;
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
            ax[i] = anchors(i, 0); ay[i] = anchors(i, 1);
            wx[i] = weights(i, 0); wy[i] = weights(i, 1);
        }
        const double a0x = poly(0, 0), axx = poly(1, 0), ayx = poly(2, 0);
        const double a0y = poly(0, 1), axy = poly(1, 1), ayy = poly(2, 1);
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
                    const double k = pmTpsKernel(ddc*ddc + ddr*ddr);
                    dx += wx[i] * k;
                    dy += wy[i] * k;
                }
                px[c] += (float)dx;
                py[c] += (float)dy;
            }
        }
    }
};

PMSimRBF pmBuildSurfaceWarp(const cv::Matx23d& M,
                            const std::vector<cv::Vec2d>& src_in,
                            const std::vector<cv::Vec2d>& resid_in)
{
    PMSimRBF S; S.M = M;
    if (src_in.empty()) { S.poly = Eigen::MatrixXd::Zero(3, 2); return S; }
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
            const double dc = src(i, 0) - src(j, 0);
            const double dr = src(i, 1) - src(j, 1);
            Aug(i, j) = pmTpsKernel(dc*dc + dr*dr);
        }
        Aug(i, N+0) = 1.0; Aug(i, N+1) = src(i, 0); Aug(i, N+2) = src(i, 1);
        Aug(N+0, i) = 1.0; Aug(N+1, i) = src(i, 0); Aug(N+2, i) = src(i, 1);
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
        const double rr = std::hypot(res(i, 0), res(i, 1));
        sse += rr * rr;
        if (rr > mx) mx = rr;
    }
    S.resid_rms = std::sqrt(sse / std::max(1, N));
    S.resid_max = mx;
    return S;
}

int pmDecimateAnchors(std::vector<cv::Vec2d>& src,
                      std::vector<cv::Vec2d>& resid, int cap)
{
    if (cap <= 0 || (int)src.size() <= cap) return (int)src.size();
    double cmin = 1e18, cmax = -1e18, rmin = 1e18, rmax = -1e18;
    for (const auto& s : src) {
        if (s[0] < cmin) cmin = s[0];
        if (s[0] > cmax) cmax = s[0];
        if (s[1] < rmin) rmin = s[1];
        if (s[1] > rmax) rmax = s[1];
    }
    const double bw = std::max(1.0, cmax - cmin + 1.0);
    const double bh = std::max(1.0, rmax - rmin + 1.0);
    double B = std::sqrt(bw * bh / std::max(1, cap));
    if (B < 1.0) B = 1.0;
    for (int it = 0; it < 8; ++it) {
        std::unordered_map<int64_t, size_t> first_in_cell;
        first_in_cell.reserve(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            const int64_t qc = (int64_t)std::floor((src[i][0] - cmin) / B);
            const int64_t qr = (int64_t)std::floor((src[i][1] - rmin) / B);
            const int64_t key = (qc << 32) ^ (qr & 0xffffffff);
            first_in_cell.emplace(key, i);
        }
        if ((int)first_in_cell.size() <= cap) {
            std::vector<cv::Vec2d> ns, nr;
            ns.reserve(first_in_cell.size());
            nr.reserve(first_in_cell.size());
            for (const auto& kv : first_in_cell) {
                ns.push_back(src[kv.second]);
                nr.push_back(resid[kv.second]);
            }
            src.swap(ns); resid.swap(nr);
            return (int)src.size();
        }
        B *= 1.25;
    }
    return (int)src.size();
}

// =============================================================================
// vc_merge_patch-specific code.
// =============================================================================

struct PMConfig {
    int      border_cells{kDefaultBorderCells};
    int      blend_cells{kDefaultBlendCells};
    int      ransac_iters{3000};
    double   ransac_min_thresh{5.0};
    double   ransac_max_thresh{10.0};
    double   ransac_mad_k{3.0};
    uint32_t ransac_seed{0};
    int      anchor_cap{0};
    int      idw_k{4};
};

struct PMSurface {
    std::string                  name;
    fs::path                     path;
    std::shared_ptr<QuadSurface> qs;
    Mat1b                        mask;
    int                          H{0}, W{0};
    int                          valid{0};
};

PMSurface pmLoadSurface(const fs::path& p)
{
    auto up = load_quad_from_tifxyz(p.string());
    if (!up) throw std::runtime_error("failed to load surface: " + p.string());
    PMSurface s;
    s.name = p.filename().string();
    s.path = p;
    s.qs = std::shared_ptr<QuadSurface>(up.release());
    const cv::Mat_<cv::Vec3f> pts = s.qs->rawPoints();
    s.H = pts.rows; s.W = pts.cols;
    s.mask = Mat1b::zeros(s.H, s.W);
    int valid = 0;
    for (int r = 0; r < s.H; ++r) {
        const cv::Vec3f* p_row = pts.ptr<cv::Vec3f>(r);
        uint8_t* mp = s.mask.ptr<uint8_t>(r);
        for (int c = 0; c < s.W; ++c) {
            const bool ok = p_row[c][0] != -1.0f;
            mp[c] = ok ? 1 : 0;
            if (ok) ++valid;
        }
    }
    s.valid = valid;
    return s;
}

// Distance to the nearest invalid/outside cell. Cells inside the child's
// valid region get a positive distance; outside cells get 0. This drives
// both the border-anchor mask and the per-cell blend weight ramp.
Mat1f pmDistToBoundary(const Mat1b& mask)
{
    cv::Mat src;
    mask.convertTo(src, CV_8U, 255.0);
    cv::Mat dist;
    cv::distanceTransform(src, dist, vc::opencv::distanceL2, cv::DIST_MASK_PRECISE);
    Mat1f out;
    dist.convertTo(out, CV_32F);
    return out;
}

Mat1b pmBorderMask(const Mat1b& mask, const Mat1f& distToBoundary, int border_cells)
{
    Mat1b out(mask.size(), (uint8_t)0);
    const float thr = (float)border_cells;
    for (int r = 0; r < mask.rows; ++r) {
        const uint8_t* mp = mask.ptr<uint8_t>(r);
        const float* dp = distToBoundary.ptr<float>(r);
        uint8_t* op = out.ptr<uint8_t>(r);
        for (int c = 0; c < mask.cols; ++c) {
            if (mp[c] && dp[c] < thr) op[c] = 1;
        }
    }
    return out;
}

struct PMAnchors {
    std::vector<Anchor> anchors;
    Mat1b               realChild;
    Mat1b               realParent;
    float               threshold{0.0f};
    double              score{0.0};
    int                 best_idx{-1};
    json                diag = json::array();
    bool                child_seeded{true};
};

// Threshold sweep over child<->parent overlap. realChild is intersected with
// the child border before scoring/anchor seeding so anchors only land on the
// rim. The fallback path (one-sided seed) preserves the symmetric handling
// from vc_merge_tifxyz for thin/asymmetric overlaps.
PMAnchors pmComputeChildAnchors(const PMSurface& child, const PMSurface& parent,
                                const SurfacePatchIndex& idx,
                                const Mat1b& child_border)
{
    PMAnchors R;
    constexpr int kNumThresholds = sizeof(kThresholds) / sizeof(kThresholds[0]);
    const float maxT = kThresholds[kNumThresholds - 1];

    OverlapMaps mAB = computeOverlap(child.qs,  parent.qs, idx, maxT);
    OverlapMaps mBA = computeOverlap(parent.qs, child.qs,  idx, maxT);

    int    bestIdx       = -1;
    double bestScore     = -1.0;
    long   bestRealCells = -1;
    Mat1b  bestRealA, bestRealB;
    int    oneIdx        = -1;
    double oneScore      = 0.0;
    long   oneRealCells  = -1;
    Mat1b  oneRealA, oneRealB;
    bool   oneSideChild  = true;

    for (int i = 0; i < kNumThresholds; ++i) {
        const float t = kThresholds[i];
        Mat1b maskA = maskAtThreshold(mAB.distance, t);
        Mat1b maskB = maskAtThreshold(mBA.distance, t);
        Mat1b realA, realB;
        Stats sA = summarize(child.qs->rawPoints(),  mAB.distance, mAB.normAgree, maskA, realA);
        Stats sB = summarize(parent.qs->rawPoints(), mBA.distance, mBA.normAgree, maskB, realB);

        // Border-only filter on the child side: anchors must come from the
        // rim, never from the interior the child is replacing.
        for (int y = 0; y < realA.rows; ++y) {
            const uint8_t* bp = child_border.ptr<uint8_t>(y);
            uint8_t* ap = realA.ptr<uint8_t>(y);
            for (int x = 0; x < realA.cols; ++x) {
                if (!bp[x]) ap[x] = 0;
            }
        }
        sA.realCells = cv::countNonZero(realA);

        const double pA = sA.matches > 0 ? double(sA.realCells)/sA.matches : 0.0;
        const double pB = sB.matches > 0 ? double(sB.realCells)/sB.matches : 0.0;
        const double score = std::sqrt(pA * pB);
        const long minReal = std::min(sA.realCells, sB.realCells);
        const long maxReal = std::max(sA.realCells, sB.realCells);
        const bool aOk = sA.realCells >= kMinComponent;
        const bool bOk = sB.realComponents >= 1 && sB.realCells >= kMinComponent;
        const bool valid = aOk && bOk;

        R.diag.push_back({
            {"threshold",          t},
            {"child_real_border",  sA.realCells},
            {"parent_real",        sB.realCells},
            {"score",              score},
            {"valid",              valid},
        });

        if (valid && (score > bestScore
                      || (score == bestScore && minReal > bestRealCells))) {
            bestScore     = score;
            bestRealCells = minReal;
            bestIdx       = i;
            bestRealA     = realA.clone();
            bestRealB     = realB.clone();
        }
        if (!valid && (aOk || bOk) && maxReal > oneRealCells) {
            oneIdx        = i;
            oneScore      = score;
            oneRealCells  = maxReal;
            oneRealA      = realA.clone();
            oneRealB      = realB.clone();
            oneSideChild  = sA.realCells >= sB.realCells;
        }
    }

    bool useChild = true;
    if (bestIdx < 0 && oneIdx >= 0) {
        bestIdx       = oneIdx;
        bestScore     = oneScore;
        bestRealCells = oneRealCells;
        bestRealA     = std::move(oneRealA);
        bestRealB     = std::move(oneRealB);
        useChild      = oneSideChild;
        std::cout << "  one-sided fallback: anchors from "
                  << (useChild ? "child" : "parent") << " side ("
                  << bestRealCells << " real cells)\n";
    }
    if (bestIdx < 0) {
        throw std::runtime_error("vc_merge_patch: no valid threshold yielded "
            "enough border overlap; raise --border-cells or check that the "
            "child border actually overlaps the parent");
    }

    const float bestT = kThresholds[bestIdx];
    std::vector<Anchor> anchors;
    if (useChild) {
        anchors = pickAnchors(child.qs, parent.qs, idx, bestRealA,
                              mAB.distance, mAB.normAgree,
                              kAnchorBinSize, bestT);
    } else {
        anchors = pickAnchors(parent.qs, child.qs, idx, bestRealB,
                              mBA.distance, mBA.normAgree,
                              kAnchorBinSize, bestT);
        // pickAnchors fills a_* with the seed-side coords. We seeded from
        // the parent; swap so the Anchor convention (a_*=child, b_*=parent)
        // holds for the remainder of the pipeline.
        for (auto& a : anchors) {
            std::swap(a.a_row,   a.b_row);
            std::swap(a.a_col,   a.b_col);
            std::swap(a.a_world, a.b_world);
        }
    }

    R.anchors      = std::move(anchors);
    R.realChild    = std::move(bestRealA);
    R.realParent   = std::move(bestRealB);
    R.threshold    = bestT;
    R.score        = bestScore;
    R.best_idx     = bestIdx;
    R.child_seeded = useChild;
    return R;
}

struct PMPatchStats {
    long footprint_cells{0};
    long replaced{0};
    long blended{0};
    long parent_filled_invalid{0};
    long no_child_neighbor{0};
};

// Mutates parent_points in place. Two passes over the warped child's
// bounding box in parent grid space:
//
//   1. Footprint pass. For each parent cell in the bbox, KDTree-query the
//      nearest warped child cell. Cells whose nearest feature is within
//      `tol` (~one warped feature spacing) are flagged as in-footprint.
//   2. Patch pass. Run a distance transform on that footprint to get the
//      parent-cell distance from the warped child boundary, then for each
//      in-footprint cell: KDTree-IDW the K nearest warped child cells'
//      XYZ (Gaussian weighting), and blend against the existing parent
//      using a smoothstep ramp 0 -> 1 over the outermost `blend_cells`
//      parent cells. Cells `blend_cells` parent cells in are pure child;
//      cells right at the warped footprint boundary stay pure parent.
//
// Basing the blend on distance-from-warped-boundary (instead of the
// child's own distance-to-boundary IDW-averaged across K neighbours) is
// what fixes the sharp edge: an outer-edge parent cell whose nearest
// warped feature happens to sit a few cells in no longer gets a heavy
// "child" pull from interior neighbours.
//
// Cells where the parent is invalid are filled with pure child XYZ
// regardless of the blend ramp so a child can extend into parent holes.
PMPatchStats pmApplyPatch(cv::Mat_<cv::Vec3f>& parent_points,
                          const Mat1b& parent_mask,
                          const PMSurface& child,
                          const PMSimRBF& warp,
                          int blend_cells, int idw_k)
{
    const int parent_H = parent_points.rows;
    const int parent_W = parent_points.cols;

    Mat1f warpedC, warpedR;
    warp.evalGrid(child.H, child.W, warpedC, warpedR);

    // Build KDTree on the warped child UVs of every valid child cell. The
    // feature index encodes (row, col) into the child grid so the IDW step
    // can recover the source cell. Also collect bbox + a robust spacing
    // estimate so the footprint membership tolerance scales with the warp.
    std::vector<cv::Vec2f> feats;
    std::vector<int>       feat_rc;
    feats.reserve(child.valid);
    feat_rc.reserve(child.valid);
    double cmin = 1e18, cmax = -1e18, rmin = 1e18, rmax = -1e18;
    for (int r = 0; r < child.H; ++r) {
        const uint8_t* mp = child.mask.ptr<uint8_t>(r);
        const float* uu = warpedC.ptr<float>(r);
        const float* vv = warpedR.ptr<float>(r);
        for (int c = 0; c < child.W; ++c) {
            if (!mp[c]) continue;
            if (!std::isfinite(uu[c]) || !std::isfinite(vv[c])) continue;
            feats.emplace_back(uu[c], vv[c]);
            feat_rc.push_back(r * child.W + c);
            if (uu[c] < cmin) cmin = uu[c];
            if (uu[c] > cmax) cmax = uu[c];
            if (vv[c] < rmin) rmin = vv[c];
            if (vv[c] > rmax) rmax = vv[c];
        }
    }
    if (feats.empty())
        throw std::runtime_error("vc_merge_patch: no valid child cells warped into parent");

    cv::Mat featMat((int)feats.size(), 2, CV_32F);
    for (size_t i = 0; i < feats.size(); ++i) {
        featMat.at<float>((int)i, 0) = feats[i][0];
        featMat.at<float>((int)i, 1) = feats[i][1];
    }
    cv::flann::Index tree(featMat, cv::flann::KDTreeIndexParams(1));

    cv::Mat_<cv::Vec3f> child_pts = child.qs->rawPoints();

    PMPatchStats stats;

    // Warp scale: ratio of warped child linear distance to native child
    // distance. With scale > 1 (dense child) the warped child cells sit at
    // ~`scale` parent cells apart, so the interpolation kernel width and
    // the footprint-membership tolerance must scale up too -- otherwise
    // some parent cells fall in the gaps between features and others sit
    // on top of a feature, producing the bouncing.
    const double M00 = warp.M(0,0), M01 = warp.M(0,1);
    const double M10 = warp.M(1,0), M11 = warp.M(1,1);
    const double det = std::abs(M00*M11 - M01*M10);
    const double scale_est = std::max(1e-6, std::sqrt(det));

    // Gaussian kernel half-width tracks one feature spacing; tolerance is
    // ~one feature spacing + sqrt(2)/2 so an interior parent cell at the
    // worst-case center of a quad still finds neighbours.
    const double sigma  = std::max(0.7, scale_est * 0.7);
    const double sigma2 = sigma * sigma;
    const float  tol    = (float)std::max(1.5, scale_est * 1.5);

    const int xmin = std::max(0,            (int)std::floor(cmin));
    const int xmax = std::min(parent_W - 1, (int)std::ceil (cmax));
    const int ymin = std::max(0,            (int)std::floor(rmin));
    const int ymax = std::min(parent_H - 1, (int)std::ceil (rmax));
    if (ymax < ymin || xmax < xmin) {
        std::cerr << "warning: warped child bbox empty in parent grid (alignment failed?)\n";
        return stats;
    }

    // K large enough to capture the ~3-sigma Gaussian support even when
    // features are at scale-spacing. Beyond-3-sigma neighbours contribute
    // < 1% weight and effectively drop out of the average.
    const int K = std::clamp(idw_k, 8, 16);

    // Pass 1: footprint mask. A parent cell is in-footprint iff some
    // warped child feature is within `tol` of it. This mask defines the
    // boundary used for the smoothstep blend ramp.
    Mat1b footprint = Mat1b::zeros(parent_H, parent_W);
    for (int r = ymin; r <= ymax; ++r) {
        for (int c = xmin; c <= xmax; ++c) {
            cv::Mat q(1, 2, CV_32F);
            q.at<float>(0, 0) = (float)c;
            q.at<float>(0, 1) = (float)r;
            cv::Mat ind(1, 1, CV_32S), d2(1, 1, CV_32F);
            tree.knnSearch(q, ind, d2, 1, cv::flann::SearchParams(32));
            if (std::sqrt(std::max(0.f, d2.at<float>(0, 0))) <= tol)
                footprint(r, c) = 1;
        }
    }

    // distanceTransform(mask)(r,c) = euclidean distance from (r,c) to the
    // nearest zero cell. Inside the footprint this is "parent cells from
    // the warped child boundary".
    cv::Mat dt_in;
    cv::distanceTransform(footprint, dt_in, vc::opencv::distanceL2, cv::DIST_MASK_PRECISE);

    // Pass 2: actual patching. Smoothstep blend by dt_in.
    const double blend_inv = 1.0 / std::max(1, blend_cells);
    for (int r = ymin; r <= ymax; ++r) {
        for (int c = xmin; c <= xmax; ++c) {
            if (!footprint(r, c)) {
                ++stats.no_child_neighbor;
                continue;
            }
            ++stats.footprint_cells;

            cv::Mat q(1, 2, CV_32F);
            q.at<float>(0, 0) = (float)c;
            q.at<float>(0, 1) = (float)r;
            cv::Mat ind(1, K, CV_32S), d2(1, K, CV_32F);
            tree.knnSearch(q, ind, d2, K, cv::flann::SearchParams(32));

            // Gaussian-weighted IDW for child XYZ. Kernel width matches
            // one warped feature spacing so adjacent parent cells see
            // smoothly varying weight profiles regardless of whether they
            // sit on a feature or between features.
            double w[16];
            double wsum = 0.0;
            for (int k = 0; k < K; ++k) {
                const double dd2 = std::max(0.0, (double)d2.at<float>(0, k));
                w[k] = std::exp(-0.5 * dd2 / sigma2);
                wsum += w[k];
            }
            if (wsum <= 0.0) { ++stats.no_child_neighbor; continue; }
            double Xs = 0, Ys = 0, Zs = 0;
            for (int k = 0; k < K; ++k) {
                const int idx_v = ind.at<int>(0, k);
                if (idx_v < 0 || idx_v >= (int)feat_rc.size()) continue;
                const int rc_packed = feat_rc[idx_v];
                const int cr = rc_packed / child.W;
                const int cc = rc_packed % child.W;
                const cv::Vec3f& p = child_pts(cr, cc);
                Xs += w[k] * p[0];
                Ys += w[k] * p[1];
                Zs += w[k] * p[2];
            }
            Xs /= wsum; Ys /= wsum; Zs /= wsum;

            // Smoothstep blend by parent-grid distance from the warped
            // footprint boundary. blend_w = 0 right at the boundary
            // (pure parent), 1 at >= blend_cells inside (pure child),
            // with a C^1 ramp in between.
            const double dt_val = (double)dt_in.at<float>(r, c);
            const double t = std::clamp(dt_val * blend_inv, 0.0, 1.0);
            double blend_w = t * t * (3.0 - 2.0 * t);

            const bool parent_valid = parent_mask(r, c) != 0;
            if (!parent_valid) {
                blend_w = 1.0;
                ++stats.parent_filled_invalid;
            }

            cv::Vec3f& pp = parent_points(r, c);
            if (parent_valid) {
                pp[0] = (float)(blend_w * Xs + (1.0 - blend_w) * pp[0]);
                pp[1] = (float)(blend_w * Ys + (1.0 - blend_w) * pp[1]);
                pp[2] = (float)(blend_w * Zs + (1.0 - blend_w) * pp[2]);
            } else {
                pp[0] = (float)Xs; pp[1] = (float)Ys; pp[2] = (float)Zs;
            }

            if (blend_w >= 1.0 - 1e-9) ++stats.replaced;
            else                       ++stats.blended;
        }
    }
    return stats;
}

int pmRun(PMSurface parent, PMSurface child, const PMConfig& cfg)
{
    std::cout << "[0/6] parent + child ready\n";
    std::cout << "  parent: " << parent.name
              << "  shape=(" << parent.H << "," << parent.W << ")"
              << "  valid=" << parent.valid << "\n";
    std::cout << "  child : " << child.name
              << "  shape=(" << child.H  << "," << child.W  << ")"
              << "  valid=" << child.valid  << "\n";

    std::cout << "[1/6] child border (border_cells=" << cfg.border_cells << ")\n";
    Mat1f childDist = pmDistToBoundary(child.mask);
    Mat1b childBorder = pmBorderMask(child.mask, childDist, cfg.border_cells);
    const int borderCount = cv::countNonZero(childBorder);
    std::cout << "  border cells: " << borderCount << " / " << child.valid << "\n";
    if (borderCount == 0)
        throw std::runtime_error("vc_merge_patch: child border is empty; check --border-cells");

    std::cout << "[2/6] building SurfacePatchIndex\n";
    SurfacePatchIndex patchIndex;
    {
        std::vector<SurfacePatchIndex::SurfacePtr> all = { child.qs, parent.qs };
        patchIndex.rebuild(all);
    }

    std::cout << "[3/6] per-edge overlap + anchor selection (border-only)\n";
    PMAnchors pmA = pmComputeChildAnchors(child, parent, patchIndex, childBorder);
    std::cout << "  best threshold=" << pmA.threshold
              << "  score="          << std::fixed << std::setprecision(4) << pmA.score
              << "  anchors="        << pmA.anchors.size()
              << "  seeded="         << (pmA.child_seeded ? "child" : "parent") << "\n";
    std::cout.unsetf(std::ios::fixed);
    if (pmA.anchors.size() < 4)
        throw std::runtime_error("vc_merge_patch: insufficient anchors (" +
            std::to_string(pmA.anchors.size()) +
            "); raise --border-cells or check inputs");

    std::vector<cv::Vec2d> pChild, pParent;
    pChild.reserve(pmA.anchors.size());
    pParent.reserve(pmA.anchors.size());
    for (const auto& a : pmA.anchors) {
        pChild.emplace_back((double)a.a_col,  (double)a.a_row);
        pParent.emplace_back((double)a.b_col, (double)a.b_row);
    }

    PMRansac R = pmRansacSimilarity(pChild, pParent,
        cfg.ransac_iters, cfg.ransac_min_thresh, cfg.ransac_max_thresh,
        cfg.ransac_mad_k, cfg.ransac_seed);
    std::cout << "  RANSAC inliers=" << R.n_in << "/" << pChild.size()
              << "  thresh="   << std::fixed << std::setprecision(2) << R.thresh
              << "  sigma_in=" << R.sigma_in
              << "  scale="    << std::setprecision(4) << pmSimilarityScale(R.M)
              << "  t=["       << std::setprecision(2) << R.M(0,2)
              << ", "          << R.M(1,2) << "]\n";
    std::cout.unsetf(std::ios::fixed);

    std::vector<cv::Vec2d> pChild_in, pParent_in;
    pChild_in.reserve(R.n_in);
    pParent_in.reserve(R.n_in);
    for (size_t i = 0; i < pChild.size(); ++i) {
        if (R.inlier[i]) {
            pChild_in.push_back(pChild[i]);
            pParent_in.push_back(pParent[i]);
        }
    }

    std::cout << "[4/6] TPS RBF on inlier residuals"
              << (cfg.anchor_cap > 0 ? " (anchor_cap=" + std::to_string(cfg.anchor_cap) + ")" : "")
              << "\n";
    // Residual = pParent - M_child * pChild. The warp aims to land each
    // child anchor exactly on its parent partner, so the child border
    // matches the parent geometry to within the RBF residual.
    std::vector<cv::Vec2d> src = pChild_in;
    std::vector<cv::Vec2d> resid;
    resid.reserve(src.size());
    for (size_t i = 0; i < pChild_in.size(); ++i)
        resid.push_back(pParent_in[i] - pmApplyAffine(R.M, pChild_in[i]));
    const int raw_n  = (int)src.size();
    const int kept_n = pmDecimateAnchors(src, resid, cfg.anchor_cap);
    if (kept_n != raw_n)
        std::cout << "  decimated " << raw_n << " -> " << kept_n
                  << " anchors (cap=" << cfg.anchor_cap << ")\n";
    PMSimRBF warp = pmBuildSurfaceWarp(R.M, src, resid);
    std::cout << "  RBF anchors=" << warp.n_anchors
              << "  resid_rms="   << std::fixed << std::setprecision(2) << warp.resid_rms
              << "  max="         << warp.resid_max << "\n";
    std::cout.unsetf(std::ios::fixed);

    std::cout << "[5/6] applying patch (blend_cells=" << cfg.blend_cells << ")\n";
    cv::Mat_<cv::Vec3f> parent_points = parent.qs->rawPoints();
    PMPatchStats stats = pmApplyPatch(parent_points, parent.mask,
                                      child, warp, cfg.blend_cells, cfg.idw_k);
    std::cout << "  footprint_cells="       << stats.footprint_cells
              << "  replaced="              << stats.replaced
              << "  blended="               << stats.blended
              << "  parent_filled_invalid=" << stats.parent_filled_invalid
              << "  no_child_neighbor="     << stats.no_child_neighbor << "\n";

    // Overwrite the parent tifxyz in place via QuadSurface::saveOverwrite.
    // That call:
    //   1. saveSnapshot(8) -- copies the LAST-PERSISTED on-disk state
    //      into <parent_dir>/backups/<parent_name>/{0..7}/ (rotating ring,
    //      a sibling of the segment dir), same convention used by grow /
    //      brush / rotation edits. Snapshot
    //      failures are logged via the QuadSurface logger but never block
    //      the actual save.
    //   2. save(force_overwrite=true) -- writes new x/y/z/meta to a
    //      sibling temp dir, carries forward aux files (approval.tif,
    //      generations.tif, etc.) into that temp dir, then atomically
    //      swaps temp <-> parent (renameat2 RENAME_EXCHANGE on Linux,
    //      remove+rename fallback elsewhere). Cells we didn't touch
    //      retain their parent values bit-identical to the input.
    // invalidateMask + writeValidMask re-derive mask.tif from the patched
    // points so any newly valid cells are reflected.
    std::cout << "[6/6] overwriting parent tifxyz at " << parent.path
              << " (snapshot to " << parent.path.parent_path().string()
              << "/backups/" << parent.name << "/)\n";

    parent.qs->invalidateMask();
    parent.qs->saveOverwrite();
    parent.qs->writeValidMask();

    const fs::path summary_path = parent.path / (parent.name + "_summary.json");
    json summary = {
        {"parent",       parent.path.string()},
        {"child",        child.path.string()},
        {"output",       parent.path.string()},
        {"border_cells", cfg.border_cells},
        {"blend_cells",  cfg.blend_cells},
        {"anchors", {
            {"raw",              (int)pmA.anchors.size()},
            {"ransac_inliers",   R.n_in},
            {"ransac_thresh",    R.thresh},
            {"ransac_sigma_in",  R.sigma_in},
            {"best_threshold",   pmA.threshold},
            {"best_score",       pmA.score},
            {"seeded",           pmA.child_seeded ? "child" : "parent"},
        }},
        {"warp", {
            {"n_anchors", warp.n_anchors},
            {"resid_rms", warp.resid_rms},
            {"resid_max", warp.resid_max},
            {"M", {{R.M(0,0), R.M(0,1), R.M(0,2)},
                   {R.M(1,0), R.M(1,1), R.M(1,2)}}},
            {"scale",     pmSimilarityScale(R.M)},
        }},
        {"patch", {
            {"footprint_cells",       stats.footprint_cells},
            {"replaced",              stats.replaced},
            {"blended",               stats.blended},
            {"parent_filled_invalid", stats.parent_filled_invalid},
            {"no_child_neighbor",     stats.no_child_neighbor},
        }},
        {"params", {
            {"ransac_iters",      cfg.ransac_iters},
            {"ransac_min_thresh", cfg.ransac_min_thresh},
            {"ransac_max_thresh", cfg.ransac_max_thresh},
            {"ransac_mad_k",      cfg.ransac_mad_k},
            {"ransac_seed",       cfg.ransac_seed},
            {"anchor_cap",        cfg.anchor_cap},
            {"idw_k",             cfg.idw_k},
        }},
        {"diag", pmA.diag},
    };
    {
        std::ofstream f(summary_path);
        f << summary.dump(2) << std::endl;
    }
    std::cout << "wrote summary to " << summary_path << "\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    PMConfig defaults;
    po::options_description desc(
        "vc_merge_patch <tifxyz_a> <tifxyz_b> [options]\n"
        "\n"
        "Patch a small child tifxyz into a parent tifxyz. The child's border "
        "overlaps the parent and is smoothly blended into it; the child "
        "interior fully replaces the corresponding region of the parent. "
        "Cells outside the child footprint are written verbatim from the "
        "parent (no rasterization, no UV transform). The parent tifxyz is "
        "overwritten in place -- back up the parent dir first if you want "
        "to keep the pre-patch version.\n"
        "\n"
        "Pass the two tifxyz dirs as positional args; the larger of the two "
        "(by valid-cell count) is taken as the parent. Use --parent / "
        "--child if you want to set the roles explicitly (e.g. ambiguous "
        "sizes, or for scripts).\n"
        "\n"
        "Backups: before overwriting, the parent's last-persisted on-disk "
        "state is snapshotted to <parent_dir>/backups/<parent_name>/{0..7}/ "
        "(rotating 8-slot ring, same convention as grow / brush / rotation "
        "edits in VC3D)");
    desc.add_options()
        ("help,h", "print help")
        ("parent,p", po::value<std::string>(),
            "Parent tifxyz directory (explicit override). When both "
            "--parent and --child are set, auto-detection is skipped. "
            "Overwritten in place: x/y/z/mask/meta are replaced atomically; "
            "aux files in the parent dir (approval.tif, generations.tif, "
            "...) are preserved.")
        ("child,c", po::value<std::string>(),
            "Child tifxyz directory (explicit override). Its border should "
            "already overlap parent geometry; alignment refines the seam, "
            "blending smooths it, and the child interior replaces the "
            "parent there.")
        ("border-cells", po::value<int>()->default_value(defaults.border_cells),
            "Width (in child grid cells) of the rim used for anchor seeding. "
            "Only child cells whose distance-to-boundary is < border-cells "
            "can become alignment anchors, so the seam fit is driven by the "
            "child's perimeter rather than its interior. Wider rim = more "
            "anchors, narrower rim = anchors live closer to the seam itself.")
        ("blend-cells", po::value<int>()->default_value(defaults.blend_cells),
            "Width (in PARENT grid cells) of the smoothstep blend ramp at "
            "the seam. Measured from the warped child footprint boundary "
            "inward. Cells right at the boundary stay pure parent; cells "
            "blend-cells inside are pure-child override. Bigger values = "
            "softer seam, more parent influence carried into the patch.")
        ("ransac-iters", po::value<int>()->default_value(defaults.ransac_iters),
            "RANSAC similarity trial count.")
        ("ransac-min-thresh", po::value<double>()->default_value(defaults.ransac_min_thresh),
            "RANSAC inlier-distance lower bound (vox).")
        ("ransac-max-thresh", po::value<double>()->default_value(defaults.ransac_max_thresh),
            "RANSAC inlier-distance upper bound (vox).")
        ("ransac-mad-k", po::value<double>()->default_value(defaults.ransac_mad_k),
            "RANSAC MAD multiplier (clamped to [min, max] threshold).")
        ("ransac-seed", po::value<uint32_t>()->default_value(defaults.ransac_seed),
            "RANSAC RNG seed (0 = nondeterministic).")
        ("anchor-cap", po::value<int>()->default_value(defaults.anchor_cap),
            "Spatial decimation cap on the RBF anchor count. If >0 and the "
            "inlier set exceeds this count, decimate by keeping one ORIGINAL "
            "anchor per coarse cell so the dense TPS LU stays tractable. "
            "0 = no cap (default).")
        ("idw-k", po::value<int>()->default_value(defaults.idw_k),
            "Number of nearest child cells (by warped UV) to IDW-sample "
            "per parent footprint cell.");

    // Hidden positional sink so up to two bare paths land in `inputs` without
    // showing up in --help.
    po::options_description hidden;
    hidden.add_options()
        ("inputs", po::value<std::vector<std::string>>(), "tifxyz inputs (positional)");
    po::positional_options_description pos;
    pos.add("inputs", -1);
    po::options_description all_opts;
    all_opts.add(desc).add(hidden);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv)
            .options(all_opts).positional(pos).run(), vm);
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "error parsing arguments: " << e.what() << "\n" << desc << "\n";
        return EXIT_FAILURE;
    }

    const bool has_parent = vm.count("parent") > 0;
    const bool has_child  = vm.count("child")  > 0;
    const std::vector<std::string> positional = vm.count("inputs")
        ? vm["inputs"].as<std::vector<std::string>>()
        : std::vector<std::string>();
    const bool explicit_roles = has_parent && has_child;

    if (vm.count("help") ||
        (!explicit_roles && positional.size() != 2) ||
        (has_parent != has_child) ||
        (explicit_roles && !positional.empty()))
    {
        if (has_parent != has_child)
            std::cerr << "specify both --parent and --child, or neither\n";
        else if (explicit_roles && !positional.empty())
            std::cerr << "do not mix positional args with --parent/--child\n";
        else if (!vm.count("help"))
            std::cerr << "usage: vc_merge_patch <tifxyz_a> <tifxyz_b> [options]\n"
                         "   or: vc_merge_patch --parent <tifxyz> --child <tifxyz> [options]\n";
        std::cout << desc << "\n";
        return vm.count("help") ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    PMConfig cfg;
    cfg.border_cells      = vm["border-cells"].as<int>();
    cfg.blend_cells       = vm["blend-cells"].as<int>();
    cfg.ransac_iters      = vm["ransac-iters"].as<int>();
    cfg.ransac_min_thresh = vm["ransac-min-thresh"].as<double>();
    cfg.ransac_max_thresh = vm["ransac-max-thresh"].as<double>();
    cfg.ransac_mad_k      = vm["ransac-mad-k"].as<double>();
    cfg.ransac_seed       = vm["ransac-seed"].as<uint32_t>();
    cfg.anchor_cap        = vm["anchor-cap"].as<int>();
    cfg.idw_k             = vm["idw-k"].as<int>();

    fs::path path_a, path_b;
    if (explicit_roles) {
        path_a = vm["parent"].as<std::string>();
        path_b = vm["child"].as<std::string>();
    } else {
        path_a = positional[0];
        path_b = positional[1];
    }

    try {
        if (!fs::is_directory(path_a))
            throw std::runtime_error("not a directory: " + path_a.string());
        if (!fs::is_directory(path_b))
            throw std::runtime_error("not a directory: " + path_b.string());

        std::cout << "[0/6] loading " << path_a.filename().string()
                  << " and " << path_b.filename().string() << "\n";
        PMSurface a = pmLoadSurface(path_a);
        PMSurface b = pmLoadSurface(path_b);

        PMSurface parent_surf, child_surf;
        if (explicit_roles) {
            parent_surf = std::move(a);
            child_surf  = std::move(b);
        } else {
            // Auto-detect: bigger surface (by valid-cell count) is the
            // parent. The two roles are dramatically asymmetric in normal
            // use (parent millions of cells, child thousands), so a flat
            // valid-count compare is enough; flag if the ratio is small in
            // case the user passed two near-equal segments by mistake.
            if (a.valid >= b.valid) {
                parent_surf = std::move(a);
                child_surf  = std::move(b);
            } else {
                parent_surf = std::move(b);
                child_surf  = std::move(a);
            }
            std::cout << "  auto-detected parent: " << parent_surf.name
                      << " (valid=" << parent_surf.valid << ")\n";
            std::cout << "  auto-detected child : " << child_surf.name
                      << " (valid=" << child_surf.valid  << ")\n";
            if (child_surf.valid > 0 &&
                parent_surf.valid < 2 * child_surf.valid)
            {
                std::cout << "  warning: parent is only "
                          << (double)parent_surf.valid /
                             (double)child_surf.valid
                          << "x larger than child; pass --parent/--child "
                             "explicitly if this is wrong\n";
            }
        }

        return pmRun(std::move(parent_surf), std::move(child_surf), cfg);
    } catch (const std::exception& e) {
        std::cerr << "vc_merge_patch failed: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
