// Census kernel for vc_tifxyz_selfcross: does a tifxyz surface pass through
// itself? Header-only so the command-line app and the regression tests run
// the identical code; see vc_tifxyz_selfcross.cpp for the tool and
// core/test/test_tifxyz_selfcross.cpp for the fixtures.
#pragma once

#include <opencv2/core/mat.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vc_selfcross {

// ------------------------------------------------------------ the predicate
//
// Moller's interval-overlap triangle test, with the degenerate branches
// surfaced instead of collapsed into "no intersection", and the plane
// tolerance derived from the operands. The inputs are float32 coordinates
// running to ~1e4 voxels, where one ULP is already ~1e-3 voxels; a fixed
// 1e-6 tolerance would be a thousand times finer than the data's own
// representation and would classify representation noise.

constexpr double FLT_EPS = 1.1920929e-7;
constexpr double EPS_SCALE = 16.0;

// Interval overlaps at or below this are touches, not penetrations. Float32
// spacing near coordinate 5,000 is ~5e-4 voxels, so anything finer than this
// is below the input's own resolution.
constexpr double TOUCH_TOL = 1e-3;

inline double plane_eps(double mag) {
    return std::max(1e-9, EPS_SCALE * FLT_EPS * mag);
}

struct Vec3 {
    double x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(double a, double b, double c) : x(a), y(b), z(c) {}
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
};

inline double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }

struct Tri {
    Vec3 a, b, c;
    int32_t v = 0, u = 0;   // grid origin of the owning quad
    int32_t t = 0;          // triangle index within the quad (0/1)
};

enum Verdict { NONE = 0, TRANSVERSE = 1, COPLANAR = 2, GRAZING = 3 };

struct TriTriResult {
    Verdict verdict = NONE;
    double penetration = 0.0;   // Euclidean length of the shared segment
    double angle_deg = 0.0;     // between the two triangle planes
    Vec3 site{};                // midpoint of the shared segment; TRANSVERSE only
};

inline bool tri2d_overlap(const Vec3 tri1[3], const Vec3 tri2[3], const Vec3& n) {
    int drop = 0;
    double ax = std::fabs(n.x), ay = std::fabs(n.y), az = std::fabs(n.z);
    if (ay > ax && ay > az) drop = 1;
    else if (az > ax && az > ay) drop = 2;

    auto proj = [drop](const Vec3& p) -> std::pair<double, double> {
        if (drop == 0) return {p.y, p.z};
        if (drop == 1) return {p.x, p.z};
        return {p.x, p.y};
    };
    std::pair<double, double> P[3], Q[3];
    for (int i = 0; i < 3; ++i) { P[i] = proj(tri1[i]); Q[i] = proj(tri2[i]); }

    auto separated = [&](const std::pair<double, double> A[3],
                         const std::pair<double, double> B[3]) {
        for (int i = 0; i < 3; ++i) {
            int j = (i + 1) % 3;
            double ex = A[j].first - A[i].first;
            double ey = A[j].second - A[i].second;
            double refs[3];
            for (int k = 0; k < 3; ++k)
                refs[k] = -ey * (B[k].first - A[i].first)
                        + ex * (B[k].second - A[i].second);
            double selfSide = -ey * (A[(i + 2) % 3].first - A[i].first)
                            + ex * (A[(i + 2) % 3].second - A[i].second);
            bool allOpposite = true;
            for (int k = 0; k < 3; ++k)
                if (refs[k] * selfSide > -1e-12) { allOpposite = false; break; }
            if (allOpposite) return true;
        }
        return false;
    };
    if (separated(P, Q)) return false;
    if (separated(Q, P)) return false;
    return true;
}

inline TriTriResult tri_tri(const Tri& T1, const Tri& T2) {
    const Vec3 p1[3] = {T1.a, T1.b, T1.c};
    const Vec3 p2[3] = {T2.a, T2.b, T2.c};

    // Bounding-box separation first. Beyond the speedup this is load-bearing:
    // without it a triangle far away from T2 but near T2's INFINITE plane
    // would be classified grazing, which says nothing about touching.
    {
        double lo1[3] = {std::min({p1[0].x, p1[1].x, p1[2].x}),
                         std::min({p1[0].y, p1[1].y, p1[2].y}),
                         std::min({p1[0].z, p1[1].z, p1[2].z})};
        double hi1[3] = {std::max({p1[0].x, p1[1].x, p1[2].x}),
                         std::max({p1[0].y, p1[1].y, p1[2].y}),
                         std::max({p1[0].z, p1[1].z, p1[2].z})};
        double lo2[3] = {std::min({p2[0].x, p2[1].x, p2[2].x}),
                         std::min({p2[0].y, p2[1].y, p2[2].y}),
                         std::min({p2[0].z, p2[1].z, p2[2].z})};
        double hi2[3] = {std::max({p2[0].x, p2[1].x, p2[2].x}),
                         std::max({p2[0].y, p2[1].y, p2[2].y}),
                         std::max({p2[0].z, p2[1].z, p2[2].z})};
        for (int k = 0; k < 3; ++k)
            if (hi1[k] < lo2[k] - TOUCH_TOL || hi2[k] < lo1[k] - TOUCH_TOL)
                return {NONE, 0.0, 0.0, {}};
    }

    Vec3 n2 = cross(p2[1] - p2[0], p2[2] - p2[0]);
    double l2 = norm(n2);
    Vec3 n1 = cross(p1[1] - p1[0], p1[2] - p1[0]);
    double l1 = norm(n1);
    if (l1 < 1e-12 || l2 < 1e-12) return {GRAZING, 0.0, 0.0, {}};   // degenerate
    n1 = n1 * (1.0 / l1);
    n2 = n2 * (1.0 / l2);

    double d1[3], d2[3];
    for (int i = 0; i < 3; ++i) d1[i] = dot(n2, p1[i] - p2[0]);
    for (int i = 0; i < 3; ++i) d2[i] = dot(n1, p2[i] - p1[0]);

    double mag = 0.0;
    for (const Vec3& p : {p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]})
        mag = std::max(mag, std::max(std::fabs(p.x),
                       std::max(std::fabs(p.y), std::fabs(p.z))));
    const double EPS = plane_eps(mag);

    bool graze = false;
    for (int i = 0; i < 3; ++i) {
        if (std::fabs(d1[i]) < EPS) graze = true;
        if (std::fabs(d2[i]) < EPS) graze = true;
    }

    auto allSameSign = [EPS](const double d[3]) {
        return (d[0] > EPS && d[1] > EPS && d[2] > EPS)
            || (d[0] < -EPS && d[1] < -EPS && d[2] < -EPS);
    };
    if (allSameSign(d1) || allSameSign(d2)) return {NONE, 0.0, 0.0, {}};

    const double ang = std::acos(std::min(1.0, std::fabs(dot(n1, n2))))
                     * 180.0 / 3.14159265358979323846;

    bool coplanar = std::fabs(d1[0]) < EPS && std::fabs(d1[1]) < EPS
                 && std::fabs(d1[2]) < EPS;
    if (coplanar)
        return tri2d_overlap(p1, p2, n1) ? TriTriResult{COPLANAR, 0.0, ang, {}}
                                         : TriTriResult{NONE, 0.0, ang, {}};
    if (graze) return {GRAZING, 0.0, ang, {}};

    Vec3 D = cross(n1, n2);
    double ax = std::fabs(D.x), ay = std::fabs(D.y), az = std::fabs(D.z);
    int idx = (ay > ax && ay > az) ? 1 : ((az > ax && az > ay) ? 2 : 0);
    auto comp = [idx](const Vec3& p) {
        return idx == 0 ? p.x : (idx == 1 ? p.y : p.z);
    };

    // The parameter interval of one triangle along the intersection line,
    // with the 3D endpoints kept: the reported crossing site is derived from
    // the shared segment itself, not from triangle centroids, which for long
    // or skew triangles can sit voxels away from the actual intersection.
    // The parameters are the endpoints' `comp` coordinates, computed from
    // the same per-component expression as before, so verdicts, penetrations
    // and angles are unchanged by carrying the endpoints along.
    auto interval = [&](const Vec3 p[3], const double d[3],
                        double& t0, double& t1, Vec3& q0, Vec3& q1) {
        int apex = -1;
        for (int i = 0; i < 3; ++i) {
            int j = (i + 1) % 3, k = (i + 2) % 3;
            if ((d[i] > 0 && d[j] <= 0 && d[k] <= 0)
             || (d[i] < 0 && d[j] >= 0 && d[k] >= 0)) { apex = i; break; }
        }
        if (apex < 0) return false;
        int j = (apex + 1) % 3, k = (apex + 2) % 3;
        q0 = p[apex] + (p[j] - p[apex]) * (d[apex] / (d[apex] - d[j]));
        q1 = p[apex] + (p[k] - p[apex]) * (d[apex] / (d[apex] - d[k]));
        t0 = comp(q0);
        t1 = comp(q1);
        if (t0 > t1) { std::swap(t0, t1); std::swap(q0, q1); }
        return true;
    };

    double a0, a1, b0, b1;
    Vec3 A0, A1, B0, B1;
    if (!interval(p1, d1, a0, a1, A0, A1)) return {GRAZING, 0.0, ang, {}};
    if (!interval(p2, d2, b0, b1, B0, B1)) return {GRAZING, 0.0, ang, {}};

    // The overlap is a projection onto one axis, shorter than the true shared
    // segment by the direction cosine; dividing restores Euclidean length,
    // which is what "penetration" should mean to a reader. Intervals that
    // merely meet at a point are a touch, not a crossing.
    const double dlen = norm(D);
    const double dcos = dlen > 1e-12
        ? std::fabs((idx == 0 ? D.x : (idx == 1 ? D.y : D.z)) / dlen) : 1.0;
    const double proj = std::min(a1, b1) - std::max(a0, b0);
    if (proj <= 0.0) return {NONE, 0.0, ang, {}};
    const double pen = proj / std::max(dcos, 1e-9);
    if (pen <= TOUCH_TOL) return {GRAZING, pen, ang, {}};
    // Both triangles' segments lie on the one intersection line, so the
    // shared segment runs from the larger of the two starts to the smaller
    // of the two ends; its midpoint is a point both interiors contain.
    const Vec3& s0 = (a0 > b0) ? A0 : B0;
    const Vec3& s1 = (a1 < b1) ? A1 : B1;
    return {TRANSVERSE, pen, ang, (s0 + s1) * 0.5};
}

// -------------------------------------------------------------- the census

struct Contact {
    int32_t v1, u1, v2, u2;
    int32_t t1, t2;
    int verdict;
    float penetration, angle_deg;
    Vec3 site;   // on the intersection segment for transverse contacts
};

struct CensusCounts {
    size_t triangles = 0, quads_dropped = 0, pairs_tested = 0;
    size_t transverse = 0, coplanar = 0, grazing = 0;
};

inline bool valid_point(const cv::Vec3f& p) {
    return (p[0] != -1.f || p[1] != -1.f || p[2] != -1.f)
        && std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

// One pass over one triangulation. Deterministic by construction: a triangle
// pair can share several broad-phase cells, and testing it in each would
// count it once per thread that happens to visit one -- the count would move
// with scheduling. Instead every pair is owned by exactly one cell, the one
// containing the minimum corner of the two bounding boxes' overlap, and is
// tested only there. No shared state, and the answer does not depend on
// thread count. Results are sorted before returning, so equal inputs give
// byte-equal outputs.
inline CensusCounts census(const cv::Mat_<cv::Vec3f>& P, int diagonal,
                           double cell, int exclude, double maxedge,
                           int nthreads, std::vector<Contact>& out)
{
    // A nonpositive cell is not a coarser grid, it is undefined behaviour:
    // zero divides to infinity whose int conversion is UB, and a negative
    // value reverses every bucket range so triangles never share a cell and
    // a crossing surface censuses clean. Refuse rather than misreport.
    if (!(cell > 0.0) || !std::isfinite(cell))
        throw std::invalid_argument("selfcross: cell size must be a finite "
                                    "positive number of voxels");
    if (exclude < 0)
        throw std::invalid_argument("selfcross: exclude must be >= 0");
    if (nthreads <= 0) nthreads = (int)std::thread::hardware_concurrency();
    if (nthreads <= 0) nthreads = 4;

    CensusCounts counts;

    std::vector<Tri> tris;
    for (int v = 0; v + 1 < P.rows; ++v) {
        for (int u = 0; u + 1 < P.cols; ++u) {
            const cv::Vec3f& q00 = P(v, u);
            const cv::Vec3f& q10 = P(v + 1, u);
            const cv::Vec3f& q01 = P(v, u + 1);
            const cv::Vec3f& q11 = P(v + 1, u + 1);
            if (!valid_point(q00) || !valid_point(q10)
             || !valid_point(q01) || !valid_point(q11))
                continue;
            Vec3 p00{q00[0], q00[1], q00[2]}, p10{q10[0], q10[1], q10[2]};
            Vec3 p01{q01[0], q01[1], q01[2]}, p11{q11[0], q11[1], q11[2]};
            if (maxedge > 0) {
                // Not cosmetic. A grid can hold discontinuities where two
                // adjacent valid cells sit far apart in 3D; a triangle built
                // across such a gap spans many wraps and crosses everything
                // it passes through, purely because the mesh has a hole.
                double e = std::max(std::max(norm(p01 - p00), norm(p11 - p01)),
                          std::max(std::max(norm(p10 - p11), norm(p00 - p10)),
                                   std::max(norm(p11 - p00), norm(p10 - p01))));
                if (e > maxedge) { ++counts.quads_dropped; continue; }
            }
            if (diagonal == 0) {
                tris.push_back({p00, p01, p11, v, u, 0});
                tris.push_back({p00, p11, p10, v, u, 1});
            } else {
                tris.push_back({p00, p01, p10, v, u, 0});
                tris.push_back({p01, p11, p10, v, u, 1});
            }
        }
    }
    counts.triangles = tris.size();
    if (tris.empty()) return counts;

    // Uniform-grid broad phase.
    Vec3 lo{1e30, 1e30, 1e30}, hi{-1e30, -1e30, -1e30};
    for (const Tri& t : tris) {
        for (const Vec3& p : {t.a, t.b, t.c}) {
            lo.x = std::min(lo.x, p.x); hi.x = std::max(hi.x, p.x);
            lo.y = std::min(lo.y, p.y); hi.y = std::max(hi.y, p.y);
            lo.z = std::min(lo.z, p.z); hi.z = std::max(hi.z, p.z);
        }
    }
    const int nx = std::max(1, (int)((hi.x - lo.x) / cell) + 1);
    const int ny = std::max(1, (int)((hi.y - lo.y) / cell) + 1);
    auto cellIdx = [&](int i, int j, int k) {
        return ((size_t)k * ny + j) * nx + i;
    };
    std::unordered_map<size_t, std::vector<uint32_t>> buckets;
    buckets.reserve(tris.size());
    for (uint32_t ti = 0; ti < tris.size(); ++ti) {
        const Tri& t = tris[ti];
        double tx0 = std::min({t.a.x, t.b.x, t.c.x});
        double tx1 = std::max({t.a.x, t.b.x, t.c.x});
        double ty0 = std::min({t.a.y, t.b.y, t.c.y});
        double ty1 = std::max({t.a.y, t.b.y, t.c.y});
        double tz0 = std::min({t.a.z, t.b.z, t.c.z});
        double tz1 = std::max({t.a.z, t.b.z, t.c.z});
        int i0 = (int)((tx0 - lo.x) / cell), i1 = (int)((tx1 - lo.x) / cell);
        int j0 = (int)((ty0 - lo.y) / cell), j1 = (int)((ty1 - lo.y) / cell);
        int k0 = (int)((tz0 - lo.z) / cell), k1 = (int)((tz1 - lo.z) / cell);
        for (int k = k0; k <= k1; ++k)
            for (int j = j0; j <= j1; ++j)
                for (int i = i0; i <= i1; ++i)
                    buckets[cellIdx(i, j, k)].push_back(ti);
    }
    std::vector<std::pair<size_t, const std::vector<uint32_t>*>> cells;
    cells.reserve(buckets.size());
    for (auto& kv : buckets)
        if (kv.second.size() > 1) cells.emplace_back(kv.first, &kv.second);
    std::sort(cells.begin(), cells.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    auto ownerCell = [&](const Tri& A, const Tri& B) {
        double ox = std::max(std::min({A.a.x, A.b.x, A.c.x}),
                             std::min({B.a.x, B.b.x, B.c.x}));
        double oy = std::max(std::min({A.a.y, A.b.y, A.c.y}),
                             std::min({B.a.y, B.b.y, B.c.y}));
        double oz = std::max(std::min({A.a.z, A.b.z, A.c.z}),
                             std::min({B.a.z, B.b.z, B.c.z}));
        return cellIdx((int)((ox - lo.x) / cell), (int)((oy - lo.y) / cell),
                       (int)((oz - lo.z) / cell));
    };

    std::vector<std::vector<Contact>> perThread(nthreads);
    std::atomic<size_t> next{0};
    std::atomic<size_t> tested{0};

    auto worker = [&](int id) {
        auto& outv = perThread[id];
        size_t local = 0;
        for (;;) {
            size_t ci = next.fetch_add(1);
            if (ci >= cells.size()) break;
            const size_t key = cells[ci].first;
            const std::vector<uint32_t>& c = *cells[ci].second;
            for (size_t a = 0; a < c.size(); ++a) {
                for (size_t b = a + 1; b < c.size(); ++b) {
                    const Tri& T1 = tris[c[a]];
                    const Tri& T2 = tris[c[b]];
                    // Quads within Chebyshev distance `exclude` share at
                    // least one vertex at exclude = 1, so this is exactly
                    // shared-vertex/shared-edge exclusion.
                    if (std::abs(T1.v - T2.v) <= exclude
                        && std::abs(T1.u - T2.u) <= exclude) continue;
                    if (ownerCell(T1, T2) != key) continue;
                    ++local;
                    TriTriResult r = tri_tri(T1, T2);
                    if (r.verdict != NONE) {
                        // Canonical endpoint order: the lexicographically
                        // smaller (v,u,t) is side 1, so every reader sees one
                        // identity per geometric pair.
                        bool swap = std::make_tuple(T1.v, T1.u, T1.t)
                                  > std::make_tuple(T2.v, T2.u, T2.t);
                        const Tri& A = swap ? T2 : T1;
                        const Tri& B = swap ? T1 : T2;
                        outv.push_back({A.v, A.u, B.v, B.u, A.t, B.t,
                                        (int)r.verdict, (float)r.penetration,
                                        (float)r.angle_deg, r.site});
                    }
                }
            }
        }
        tested.fetch_add(local);
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < nthreads; ++t) pool.emplace_back(worker, t);
    for (auto& t : pool) t.join();
    counts.pairs_tested = tested.load();

    size_t total = 0;
    for (auto& v : perThread) total += v.size();
    out.reserve(out.size() + total);
    size_t first = out.size();
    for (auto& v : perThread) out.insert(out.end(), v.begin(), v.end());
    std::sort(out.begin() + first, out.end(),
              [](const Contact& x, const Contact& y) {
        return std::tie(x.v1, x.u1, x.t1, x.v2, x.u2, x.t2)
             < std::tie(y.v1, y.u1, y.t1, y.v2, y.u2, y.t2);
    });
    for (size_t i = first; i < out.size(); ++i) {
        if (out[i].verdict == TRANSVERSE) ++counts.transverse;
        else if (out[i].verdict == COPLANAR) ++counts.coplanar;
        else ++counts.grazing;
    }
    return counts;
}

}  // namespace vc_selfcross
