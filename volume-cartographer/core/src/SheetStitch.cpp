#include "vc/core/types/SheetStitch.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>

namespace vc::sheets
{
namespace
{
using V3 = Point;

V3 operator-(const V3& a, const V3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
double dot(const V3& a, const V3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
double norm(const V3& a) { return std::sqrt(dot(a, a)); }
V3 unit(const V3& a)
{
    const double n = norm(a);
    return n > 0 ? V3{a[0] / n, a[1] / n, a[2] / n} : V3{0, 0, 0};
}

double smoothstep(double t)
{
    t = std::clamp(t, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Jacobi eigen-decomposition of a symmetric 3x3 matrix. Returns eigenvectors as
// columns of `vec`, eigenvalues in `val` (unordered). Robust and dependency-free;
// the matrices here are tiny so the fixed sweep count is ample.
void jacobiEigen(double a[3][3], double vec[3][3], double val[3])
{
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            vec[i][j] = (i == j) ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 64; ++sweep) {
        // Largest off-diagonal magnitude.
        int p = 0, q = 1;
        double off = std::abs(a[0][1]);
        if (std::abs(a[0][2]) > off) { off = std::abs(a[0][2]); p = 0; q = 2; }
        if (std::abs(a[1][2]) > off) { off = std::abs(a[1][2]); p = 1; q = 2; }
        if (off < 1e-18)
            break;
        const double app = a[p][p], aqq = a[q][q], apq = a[p][q];
        const double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double c = std::cos(phi), s = std::sin(phi);
        for (int k = 0; k < 3; ++k) {
            const double akp = a[k][p], akq = a[k][q];
            a[k][p] = c * akp - s * akq;
            a[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < 3; ++k) {
            const double apk = a[p][k], aqk = a[q][k];
            a[p][k] = c * apk - s * aqk;
            a[q][k] = s * apk + c * aqk;
        }
        for (int k = 0; k < 3; ++k) {
            const double vkp = vec[k][p], vkq = vec[k][q];
            vec[k][p] = c * vkp - s * vkq;
            vec[k][q] = s * vkp + c * vkq;
        }
    }
    for (int i = 0; i < 3; ++i)
        val[i] = a[i][i];
}

V3 column(const double vec[3][3], int c) { return {vec[0][c], vec[1][c], vec[2][c]}; }

// Per-cell L1 distance (in grid steps) from each present footprint cell to the
// nearest absent cell or the footprint's outer edge. Absent cells get 0; present
// cells 1..feather_depth. Used for the feather weight.
std::vector<int> footprintBoundaryDistance(int64_t nu, int64_t nv, const std::vector<uint8_t>& foot)
{
    const size_t n = size_t(nu * nv);
    std::vector<int> dist(n, std::numeric_limits<int>::max());
    std::deque<size_t> q;
    auto idx = [nv](int64_t i, int64_t j) { return size_t(i * nv + j); };
    for (int64_t i = 0; i < nu; ++i) {
        for (int64_t j = 0; j < nv; ++j) {
            const size_t k = idx(i, j);
            if (!foot[k]) {
                dist[k] = 0;
                q.push_back(k);
            } else if (i == 0 || j == 0 || i == nu - 1 || j == nv - 1) {
                // A present cell on the grid border abuts the (implicit) outside.
                dist[k] = 1;
                q.push_back(k);
            }
        }
    }
    while (!q.empty()) {
        const size_t k = q.front();
        q.pop_front();
        const int64_t i = int64_t(k) / nv, j = int64_t(k) % nv;
        const int d = dist[k];
        const int64_t di[4] = {-1, 1, 0, 0}, dj[4] = {0, 0, -1, 1};
        for (int t = 0; t < 4; ++t) {
            const int64_t ni = i + di[t], nj = j + dj[t];
            if (ni < 0 || nj < 0 || ni >= nu || nj >= nv)
                continue;
            const size_t nk = idx(ni, nj);
            if (foot[nk] && dist[nk] > d + 1) {
                dist[nk] = d + 1;
                q.push_back(nk);
            }
        }
    }
    return dist;
}

// Vertex index of each present footprint cell: verts are stored in row-major scan
// order (i outer, j inner) over present cells (see SheetCollection decode).
std::vector<int64_t> cellVertexIndex(int64_t nu, int64_t nv, const std::vector<uint8_t>& foot)
{
    std::vector<int64_t> idxOf(size_t(nu * nv), -1);
    int64_t next = 0;
    for (size_t k = 0; k < idxOf.size(); ++k)
        if (foot[k])
            idxOf[k] = next++;
    return idxOf;
}

// Median of a scratch vector (mutated).
double median(std::vector<double>& v)
{
    if (v.empty())
        return 0.0;
    const size_t m = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + m, v.end());
    return v[m];
}
}  // namespace

StitchResult stitchSheets(const std::vector<Mesh>& sheets, const StitchParams& params)
{
    StitchResult r;
    r.inputSheets = int(sheets.size());

    // Pass 1: centroid of all valid vertices.
    V3 sum{0, 0, 0};
    int64_t count = 0;
    for (const auto& m : sheets)
        for (const auto& v : m.verts) {
            sum[0] += v[0]; sum[1] += v[1]; sum[2] += v[2];
            ++count;
        }
    r.inputVerts = count;
    if (count < 3)
        return r;  // ok() == false
    const V3 c{sum[0] / count, sum[1] / count, sum[2] / count};

    // PCA frame: covariance of (v - centroid) -> e1, e2 span the sheet plane (largest
    // two eigenvalues), n is the surface normal (smallest).
    double cov[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (const auto& m : sheets)
        for (const auto& v : m.verts) {
            const V3 d = v - c;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    cov[i][j] += d[i] * d[j];
        }
    double vec[3][3], val[3];
    jacobiEigen(cov, vec, val);
    int order[3] = {0, 1, 2};
    std::sort(order, order + 3, [&](int a, int b) { return val[a] > val[b]; });
    const V3 e1 = unit(column(vec, order[0]));
    const V3 e2 = unit(column(vec, order[1]));
    const V3 nrm = unit(column(vec, order[2]));

    // Grid spacing: native sheet spacing unless overridden. Median adjacent-vertex
    // distance over the present footprint of each sheet.
    double cell = params.cell;
    if (cell <= 0.0) {
        std::vector<double> spacings;
        for (const auto& m : sheets) {
            const auto idxOf = cellVertexIndex(m.nu, m.nv, m.foot);
            auto vid = [&](int64_t i, int64_t j) { return idxOf[size_t(i * m.nv + j)]; };
            for (int64_t i = 0; i < m.nu; ++i)
                for (int64_t j = 0; j < m.nv; ++j) {
                    const int64_t a = vid(i, j);
                    if (a < 0)
                        continue;
                    if (j + 1 < m.nv) {
                        const int64_t b = vid(i, j + 1);
                        if (b >= 0) spacings.push_back(norm(m.verts[size_t(a)] - m.verts[size_t(b)]));
                    }
                    if (i + 1 < m.nu) {
                        const int64_t b = vid(i + 1, j);
                        if (b >= 0) spacings.push_back(norm(m.verts[size_t(a)] - m.verts[size_t(b)]));
                    }
                }
        }
        cell = median(spacings);
    }
    if (!(cell > 0.0))
        return r;  // degenerate (coincident vertices)

    // Grid extent in the (e1, e2) plane.
    double umin = 1e300, umax = -1e300, vmin = 1e300, vmax = -1e300;
    for (const auto& m : sheets)
        for (const auto& v : m.verts) {
            const V3 d = v - c;
            const double u = dot(d, e1), w = dot(d, e2);
            umin = std::min(umin, u); umax = std::max(umax, u);
            vmin = std::min(vmin, w); vmax = std::max(vmax, w);
        }
    // Auto-coarsen so the grid stays bounded. A pathological spacing (near-duplicate
    // vertices -> tiny cell) or a large union would otherwise allocate several GB of
    // accumulators per re-blend and OOM the UI. If the natural grid is too big we grow
    // the cell to fit the budget; the preview just gets coarser. ~2M nodes keeps the
    // transient accumulators (~70 B/node) around ~140 MB.
    constexpr long long kMaxNodes = 2LL * 1024 * 1024;
    {
        const double estCols = (umax - umin) / cell + 2 * params.margin + 2;
        const double estRows = (vmax - vmin) / cell + 2 * params.margin + 2;
        const double estNodes = estCols * estRows;
        if (estNodes > double(kMaxNodes))
            cell *= std::sqrt(estNodes / double(kMaxNodes));
    }
    const double u0 = umin - params.margin * cell;
    const double v0 = vmin - params.margin * cell;
    const long cols = long(std::ceil((umax - u0) / cell + params.margin)) + 1;
    const long rows = long(std::ceil((vmax - v0) / cell + params.margin)) + 1;
    if (cols < 1 || rows < 1 || (long long)cols * rows > kMaxNodes + 16)
        return r;  // empty or (after coarsening) still implausible

    r.cols = int(cols);
    r.rows = int(rows);
    r.cell = cell;
    r.origin = {c[0] + u0 * e1[0] + v0 * e2[0], c[1] + u0 * e1[1] + v0 * e2[1],
                c[2] + u0 * e1[2] + v0 * e2[2]};
    r.e1 = e1; r.e2 = e2; r.n = nrm;

    const size_t N = size_t(rows) * cols;
    std::vector<double> sumW(N, 0.0);
    std::vector<std::array<double, 3>> sumP(N, {0.0, 0.0, 0.0});
    std::vector<double> hmin(N, 1e300), hmax(N, -1e300);

    // Pass 2: feathered bilinear splat of every vertex.
    for (const auto& m : sheets) {
        const auto idxOf = cellVertexIndex(m.nu, m.nv, m.foot);
        const auto bdist = footprintBoundaryDistance(m.nu, m.nv, m.foot);
        for (int64_t i = 0; i < m.nu; ++i)
            for (int64_t j = 0; j < m.nv; ++j) {
                const int64_t a = idxOf[size_t(i * m.nv + j)];
                if (a < 0)
                    continue;
                const V3 p = m.verts[size_t(a)];
                const V3 d = p - c;
                const double u = dot(d, e1), w = dot(d, e2), h = dot(d, nrm);
                const double fw = smoothstep(double(bdist[size_t(i * m.nv + j)]) / std::max(1.0, params.feather));
                if (fw <= 0.0)
                    continue;
                const double pu = (u - u0) / cell, pv = (w - v0) / cell;
                const long c0 = long(std::floor(pu)), r0 = long(std::floor(pv));
                const double fx = pu - c0, fy = pv - r0;
                const double bw[2][2] = {{(1 - fx) * (1 - fy), fx * (1 - fy)},
                                         {(1 - fx) * fy, fx * fy}};
                for (int dr = 0; dr < 2; ++dr)
                    for (int dc = 0; dc < 2; ++dc) {
                        const long rr = r0 + dr, cc = c0 + dc;
                        if (rr < 0 || cc < 0 || rr >= rows || cc >= cols)
                            continue;
                        const double weight = fw * bw[dr][dc];
                        if (weight <= 0.0)
                            continue;
                        const size_t k = size_t(rr) * cols + cc;
                        sumW[k] += weight;
                        sumP[k][0] += weight * p[0];
                        sumP[k][1] += weight * p[1];
                        sumP[k][2] += weight * p[2];
                        hmin[k] = std::min(hmin[k], h);
                        hmax[k] = std::max(hmax[k], h);
                    }
            }
    }

    // Finalise: blended position per node; flag folds (large spread along n).
    r.grid.assign(N, Point{-1, -1, -1});
    r.valid.assign(N, 0);
    const double foldThresh = 1.5 * cell;
    for (size_t k = 0; k < N; ++k) {
        if (sumW[k] <= 1e-9)
            continue;
        r.grid[k] = {sumP[k][0] / sumW[k], sumP[k][1] / sumW[k], sumP[k][2] / sumW[k]};
        r.valid[k] = 1;
        ++r.filledCells;
        const double spread = hmax[k] - hmin[k];
        r.maxSpreadVox = std::max(r.maxSpreadVox, spread);
        if (spread > foldThresh)
            ++r.foldCells;
    }
    return r;
}

StitchResult interpolateGaps(const StitchResult& in, const InterpParams& params)
{
    StitchResult r = in;
    const int R = r.rows, C = r.cols;
    const size_t N = size_t(R) * C;
    if (N == 0 || r.filledCells == 0)
        return r;

    const std::vector<uint8_t> orig = r.valid;  // pinned boundary (true sheet data)
    auto lerp = [](const Point& a, const Point& b, double t) {
        return Point{a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t};
    };

    // Phase 1: scan-line interpolation. For each hole, find the nearest ORIGINAL sheet
    // cells bounding it along its row and along its column and linearly interpolate
    // across. A hole fills only when bounded on BOTH sides (row or column), so gaps of
    // any width bridge but the outer boundary (data on one side only) is never
    // extrapolated. Row and column interpolants are averaged where both exist.
    std::vector<int> left(N, -1), right(N, -1), up(N, -1), down(N, -1);
    for (int y = 0; y < R; ++y) {           // nearest original-valid to the left / right in each row
        int last = -1;
        for (int x = 0; x < C; ++x) { const size_t k = size_t(y) * C + x; left[k] = last; if (orig[k]) last = x; }
        last = -1;
        for (int x = C - 1; x >= 0; --x) { const size_t k = size_t(y) * C + x; right[k] = last; if (orig[k]) last = x; }
    }
    for (int x = 0; x < C; ++x) {            // nearest original-valid above / below in each column
        int last = -1;
        for (int y = 0; y < R; ++y) { const size_t k = size_t(y) * C + x; up[k] = last; if (orig[k]) last = y; }
        last = -1;
        for (int y = R - 1; y >= 0; --y) { const size_t k = size_t(y) * C + x; down[k] = last; if (orig[k]) last = y; }
    }
    for (int y = 0; y < R; ++y)
        for (int x = 0; x < C; ++x) {
            const size_t k = size_t(y) * C + x;
            if (orig[k])
                continue;
            Point acc{0, 0, 0};
            int w = 0;
            if (left[k] >= 0 && right[k] >= 0) {  // bounded along the row
                const double t = double(x - left[k]) / double(right[k] - left[k]);
                const Point p = lerp(r.grid[size_t(y) * C + left[k]], r.grid[size_t(y) * C + right[k]], t);
                acc[0] += p[0]; acc[1] += p[1]; acc[2] += p[2]; ++w;
            }
            if (up[k] >= 0 && down[k] >= 0) {     // bounded along the column
                const double t = double(y - up[k]) / double(down[k] - up[k]);
                const Point p = lerp(r.grid[size_t(up[k]) * C + x], r.grid[size_t(down[k]) * C + x], t);
                acc[0] += p[0]; acc[1] += p[1]; acc[2] += p[2]; ++w;
            }
            if (w) {
                r.grid[k] = {acc[0] / w, acc[1] / w, acc[2] / w};
                r.valid[k] = 1;
            }
        }

    auto at = [&](int y, int x) -> const Point* {
        if (y < 0 || x < 0 || y >= R || x >= C)
            return nullptr;
        const size_t k = size_t(y) * C + x;
        return r.valid[k] ? &r.grid[k] : nullptr;
    };

    // Phase 2: Laplace-smooth the filled cells (original sheet cells stay pinned) so the
    // bridge is a seam-free membrane rather than a blend of two inward fronts.
    for (int iter = 0; iter < params.maxIters; ++iter) {
        std::vector<std::pair<size_t, Point>> upd;
        double maxMove = 0.0;
        for (int y = 0; y < R; ++y)
            for (int x = 0; x < C; ++x) {
                const size_t k = size_t(y) * C + x;
                if (!r.valid[k] || orig[k])
                    continue;  // only relax newly filled cells
                Point s{0, 0, 0};
                int n = 0;
                for (const Point* p : {at(y, x - 1), at(y, x + 1), at(y - 1, x), at(y + 1, x)})
                    if (p) { s[0] += (*p)[0]; s[1] += (*p)[1]; s[2] += (*p)[2]; ++n; }
                if (!n)
                    continue;
                const Point np{s[0] / n, s[1] / n, s[2] / n};
                const double d = std::sqrt((np[0] - r.grid[k][0]) * (np[0] - r.grid[k][0]) +
                                           (np[1] - r.grid[k][1]) * (np[1] - r.grid[k][1]) +
                                           (np[2] - r.grid[k][2]) * (np[2] - r.grid[k][2]));
                maxMove = std::max(maxMove, d);
                upd.push_back({k, np});
            }
        for (const auto& [k, pt] : upd)
            r.grid[k] = pt;
        if (upd.empty() || maxMove < params.tol * r.cell)
            break;
    }

    r.filledCells = 0;
    for (size_t k = 0; k < N; ++k)
        if (r.valid[k])
            ++r.filledCells;
    return r;
}

}  // namespace vc::sheets
