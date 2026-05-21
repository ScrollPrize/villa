#include "vc/core/util/InpaintSurface.hpp"

#include <ceres/ceres.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <vector>

namespace vc::core::util {

namespace {

inline bool isValid(const cv::Vec3f& p)
{
    return p[0] != -1.f
        && std::isfinite(p[0])
        && std::isfinite(p[1])
        && std::isfinite(p[2]);
}

// 4-neighbor mean fill: cheap initial guess so the Ceres solve starts from a
// reasonable point and DistLoss isn't seeded at a singularity.
void diffuseFill(cv::Mat_<cv::Vec3d>& pts, cv::Mat_<uchar>& unknown)
{
    const int rows = pts.rows;
    const int cols = pts.cols;
    for (int iter = 0; iter < 64; ++iter) {
        cv::Mat_<cv::Vec3d> next = pts.clone();
        cv::Mat_<uchar> next_unknown = unknown.clone();
        int changed = 0;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (!unknown(r, c)) continue;
                cv::Vec3d sum(0, 0, 0);
                int n = 0;
                if (r > 0 && !unknown(r - 1, c)) { sum += pts(r - 1, c); ++n; }
                if (r + 1 < rows && !unknown(r + 1, c)) { sum += pts(r + 1, c); ++n; }
                if (c > 0 && !unknown(r, c - 1)) { sum += pts(r, c - 1); ++n; }
                if (c + 1 < cols && !unknown(r, c + 1)) { sum += pts(r, c + 1); ++n; }
                if (n > 0) {
                    next(r, c) = sum * (1.0 / n);
                    next_unknown(r, c) = 0;
                    ++changed;
                }
            }
        }
        pts = std::move(next);
        unknown = std::move(next_unknown);
        if (changed == 0) break;
    }
}

// 4-neighbor edge-length preservation: ||a - b|| should equal _d.
struct DistLoss {
    DistLoss(double d, double w) : _d(d), _w(w) {}
    template <typename T>
    bool operator()(const T* const a, const T* const b, T* residual) const
    {
        T d[3] = { a[0] - b[0], a[1] - b[1], a[2] - b[2] };
        T dist_sq = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
        if (dist_sq <= T(0)) {
            residual[0] = T(_w) * (d[0]*d[0] + d[1]*d[1] + d[2]*d[2] - T(1));
            return true;
        }
        const T dist = sqrt(dist_sq);
        const T d_sq = T(_d) * T(_d);
        if (dist_sq < d_sq) residual[0] = T(_w) * (T(_d) / dist - T(1));
        else                residual[0] = T(_w) * (dist / T(_d) - T(1));
        return true;
    }
    static ceres::CostFunction* Create(double d, double w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<DistLoss, 1, 3, 3>(new DistLoss(d, w));
    }
    double _d, _w;
};

// Collinearity penalty for a triple (a, b, c) along a grid line.
struct StraightLoss {
    explicit StraightLoss(double w) : _w(w) {}
    template <typename T>
    bool operator()(const T* const a, const T* const b, const T* const c, T* residual) const
    {
        T d1[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
        T d2[3] = { c[0] - b[0], c[1] - b[1], c[2] - b[2] };
        T l1_sq = d1[0]*d1[0] + d1[1]*d1[1] + d1[2]*d1[2];
        T l2_sq = d2[0]*d2[0] + d2[1]*d2[1] + d2[2]*d2[2];
        if (l1_sq <= T(1e-24) || l2_sq <= T(1e-24)) { residual[0] = T(0); return true; }
        T dot = (d1[0]*d2[0] + d1[1]*d2[1] + d1[2]*d2[2]) / (sqrt(l1_sq) * sqrt(l2_sq));
        residual[0] = T(_w) * (T(1) - dot);
        return true;
    }
    static ceres::CostFunction* Create(double w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<StraightLoss, 1, 3, 3, 3>(new StraightLoss(w));
    }
    double _w;
};

// Solve a single ROI: cells with unknown==1 are optimized; the rest are fixed.
void solveRoi(cv::Mat_<cv::Vec3d>& pts, const cv::Mat_<uchar>& unknown,
              double unit, int max_iters)
{
    const int rows = pts.rows;
    const int cols = pts.cols;
    ceres::Problem problem;

    auto addParam = [&](int r, int c) {
        problem.AddParameterBlock(&pts(r, c)[0], 3);
    };

    // 4-neighbor DistLoss
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (c + 1 < cols) {
                addParam(r, c); addParam(r, c + 1);
                problem.AddResidualBlock(DistLoss::Create(unit, 1.0), nullptr,
                                         &pts(r, c)[0], &pts(r, c + 1)[0]);
            }
            if (r + 1 < rows) {
                addParam(r, c); addParam(r + 1, c);
                problem.AddResidualBlock(DistLoss::Create(unit, 1.0), nullptr,
                                         &pts(r, c)[0], &pts(r + 1, c)[0]);
            }
        }
    }

    // Collinearity on row and column triples
    for (int r = 0; r < rows; ++r) {
        for (int c = 1; c + 1 < cols; ++c) {
            problem.AddResidualBlock(StraightLoss::Create(1.0), nullptr,
                                     &pts(r, c - 1)[0], &pts(r, c)[0], &pts(r, c + 1)[0]);
        }
    }
    for (int c = 0; c < cols; ++c) {
        for (int r = 1; r + 1 < rows; ++r) {
            problem.AddResidualBlock(StraightLoss::Create(1.0), nullptr,
                                     &pts(r - 1, c)[0], &pts(r, c)[0], &pts(r + 1, c)[0]);
        }
    }

    // Fix everything that wasn't originally invalid
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (!unknown(r, c) && problem.HasParameterBlock(&pts(r, c)[0])) {
                problem.SetParameterBlockConstant(&pts(r, c)[0]);
            }
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = max_iters;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}

} // namespace

int inpaintSurfaceHoles(cv::Mat_<cv::Vec3f>& points, double unit, int max_iters)
{
    const int rows = points.rows;
    const int cols = points.cols;
    if (rows <= 0 || cols <= 0) return 0;

    // 1. Build invalid mask.
    cv::Mat_<uchar> invalid(rows, cols, uchar(0));
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (!isValid(points(r, c))) invalid(r, c) = 1;
        }
    }

    // 2. Connected components of invalid cells.
    cv::Mat labels;
    const int n_components = cv::connectedComponents(invalid, labels, 8, CV_32S);
    if (n_components <= 1) return 0;

    // 3. For each interior component, compute its dilated bbox and solve.
    int total_filled = 0;
    const int dilate = 2; // need >=2 ring so StraightLoss triples have anchors
    for (int comp = 1; comp < n_components; ++comp) {
        // bbox + border-touch test in one pass
        int r0 = rows, r1 = -1, c0 = cols, c1 = -1;
        bool on_border = false;
        for (int r = 0; r < rows; ++r) {
            const int* lr = labels.ptr<int>(r);
            for (int c = 0; c < cols; ++c) {
                if (lr[c] != comp) continue;
                if (r == 0 || r == rows - 1 || c == 0 || c == cols - 1) on_border = true;
                if (r < r0) r0 = r; if (r > r1) r1 = r;
                if (c < c0) c0 = c; if (c > c1) c1 = c;
            }
        }
        if (on_border || r1 < 0) continue; // skip outer padding

        const int rr0 = std::max(0, r0 - dilate);
        const int rr1 = std::min(rows - 1, r1 + dilate);
        const int cc0 = std::max(0, c0 - dilate);
        const int cc1 = std::min(cols - 1, c1 + dilate);
        const int H = rr1 - rr0 + 1;
        const int W = cc1 - cc0 + 1;

        // 4. Extract ROI as double, mark unknowns (= all invalid cells inside).
        cv::Mat_<cv::Vec3d> roi(H, W);
        cv::Mat_<uchar> unknown(H, W, uchar(0));
        int n_unknown = 0;
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < W; ++c) {
                const cv::Vec3f& src = points(rr0 + r, cc0 + c);
                if (invalid(rr0 + r, cc0 + c)) {
                    roi(r, c) = cv::Vec3d(0, 0, 0);
                    unknown(r, c) = 1;
                    ++n_unknown;
                } else {
                    roi(r, c) = cv::Vec3d(src[0], src[1], src[2]);
                }
            }
        }
        if (n_unknown == 0) continue;

        // 5. Seed unknowns with diffusion, then refine with Ceres.
        cv::Mat_<uchar> diffuse_unknown = unknown.clone();
        diffuseFill(roi, diffuse_unknown);
        // If diffuse left any unfilled (no valid neighbor reached) skip; means
        // the component is fully detached from the ROI ring after dilation,
        // which shouldn't happen for interior holes.
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < W; ++c) {
                if (diffuse_unknown(r, c)) {
                    // fallback: leave as zero; solver will pull it via DistLoss
                }
            }
        }

        solveRoi(roi, unknown, unit, max_iters);

        // 6. Write back the inpainted cells.
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < W; ++c) {
                if (!unknown(r, c)) continue;
                const cv::Vec3d& v = roi(r, c);
                points(rr0 + r, cc0 + c) = cv::Vec3f(
                    static_cast<float>(v[0]),
                    static_cast<float>(v[1]),
                    static_cast<float>(v[2]));
                ++total_filled;
            }
        }
    }
    return total_filled;
}

} // namespace vc::core::util
