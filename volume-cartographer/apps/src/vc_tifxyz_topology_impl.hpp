#pragma once

#include "vc/core/util/SurfaceArea.hpp"

#include <opencv2/core/mat.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace vc_topology {

inline bool valid_point(const cv::Vec3f& p) {
    return (p[0] != -1.f || p[1] != -1.f || p[2] != -1.f)
        && std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

struct Vec3 {
    double x = 0, y = 0, z = 0;
};

struct Params {
    double tear_factor = 4.0;
    int max_sites = 200;
};

struct Component {
    int64_t quads = 0;
    double area_vx2 = 0.0;
    int v = 0, u = 0;
    Vec3 site{};
};

struct Hole {
    int64_t cells = 0;
    int v0 = 0, u0 = 0, v1 = 0, u1 = 0;
    int v = 0, u = 0;
    Vec3 site{};
};

struct TearSite {
    int64_t edges = 0;
    double max_len = 0.0;
    double max_ratio = 0.0;
    int v = 0, u = 0;
    Vec3 site{};
};

constexpr double FOLD_COS_TOL = 0.5;

struct FoldSite {
    int64_t quads = 0;
    int v = 0, u = 0;
    Vec3 site{};
};

struct Census {
    int rows = 0, cols = 0;
    int64_t valid_vertices = 0;
    int64_t valid_quads = 0;
    int64_t isolated_vertices = 0;
    double area_vx2 = 0.0;
    double median_step_u = 0.0;
    double median_step_v = 0.0;

    int64_t component_count = 0;
    int64_t island_quads = 0;
    double island_area_vx2 = 0.0;
    std::vector<Component> components;

    int64_t hole_count = 0;
    int64_t hole_cells = 0;
    std::vector<Hole> holes;

    int64_t tear_edges = 0;
    int64_t tear_site_count = 0;
    double max_edge_ratio = 0.0;
    std::vector<TearSite> tears;

    int64_t fold_quads = 0;
    int64_t fold_quads_other_diagonal = 0;
    int64_t fold_site_count = 0;
    int64_t degenerate_quads = 0;
    std::vector<FoldSite> folds;
};

namespace detail {

inline Vec3 vec(const cv::Vec3f& p) {
    return {(double)p[0], (double)p[1], (double)p[2]};
}

inline Vec3 sub(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

inline double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline double dist(const Vec3& a, const Vec3& b) {
    const Vec3 d = sub(a, b);
    return std::sqrt(dot(d, d));
}

struct DisjointSet {
    std::vector<int32_t> parent;
    std::vector<int32_t> size;

    explicit DisjointSet(size_t n) : parent(n), size(n, 1) {
        std::iota(parent.begin(), parent.end(), 0);
    }
    int32_t find(int32_t a) {
        while (parent[a] != a) {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        return a;
    }
    void unite(int32_t a, int32_t b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (size[a] < size[b]) std::swap(a, b);
        parent[b] = a;
        size[a] += size[b];
    }
};

inline bool opposed(const Vec3& a, const Vec3& b) {
    const double la = std::sqrt(dot(a, a)), lb = std::sqrt(dot(b, b));
    if (!(la > 0.0) || !(lb > 0.0)) return false;
    return dot(a, b) / (la * lb) < -FOLD_COS_TOL;
}

inline double median_of(std::vector<float>& v) {
    if (v.empty()) return 0.0;
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + (ptrdiff_t)mid, v.end());
    return (double)v[mid];
}

}  // namespace detail

inline Census census(const cv::Mat_<cv::Vec3f>& P, const Params& params)
{
    if (!(params.tear_factor > 1.0) || !std::isfinite(params.tear_factor))
        throw std::invalid_argument(
            "topology: tear factor must be a finite number greater than 1; "
            "at or below 1 every edge at the median step is a tear");
    if (params.max_sites < 0)
        throw std::invalid_argument("topology: max sites must be >= 0");

    using detail::dist;
    using detail::vec;

    Census c;
    c.rows = P.rows;
    c.cols = P.cols;
    if (P.empty()) return c;

    const int rows = P.rows, cols = P.cols;
    std::vector<uint8_t> vertex_valid((size_t)rows * cols, 0);
    for (int v = 0; v < rows; ++v)
        for (int u = 0; u < cols; ++u)
            if (valid_point(P(v, u))) {
                vertex_valid[(size_t)v * cols + u] = 1;
                ++c.valid_vertices;
            }

    const int qrows = std::max(0, rows - 1), qcols = std::max(0, cols - 1);
    std::vector<uint8_t> quad_valid((size_t)qrows * qcols, 0);
    std::vector<uint8_t> vertex_in_quad((size_t)rows * cols, 0);
    std::vector<uint8_t> quad_folded((size_t)qrows * qcols, 0);
    std::vector<double> quad_area((size_t)qrows * qcols, 0.0);

    for (int v = 0; v < qrows; ++v) {
        for (int u = 0; u < qcols; ++u) {
            const size_t qi = (size_t)v * qcols + u;
            const cv::Vec3f& a00 = P(v, u);
            const cv::Vec3f& a01 = P(v, u + 1);
            const cv::Vec3f& a10 = P(v + 1, u);
            const cv::Vec3f& a11 = P(v + 1, u + 1);
            if (!valid_point(a00) || !valid_point(a01) || !valid_point(a10)
                || !valid_point(a11))
                continue;
            quad_valid[qi] = 1;
            ++c.valid_quads;
            vertex_in_quad[(size_t)v * cols + u] = 1;
            vertex_in_quad[(size_t)v * cols + u + 1] = 1;
            vertex_in_quad[(size_t)(v + 1) * cols + u] = 1;
            vertex_in_quad[(size_t)(v + 1) * cols + u + 1] = 1;

            const double area = vc::surface::quadAreaVox2(a00, a10, a01, a11);
            quad_area[qi] = area;
            c.area_vx2 += area;
            if (area <= 0.0) {
                ++c.degenerate_quads;
                continue;
            }

            const Vec3 p00 = vec(a00), p01 = vec(a01);
            const Vec3 p10 = vec(a10), p11 = vec(a11);
            const Vec3 n1 = detail::cross(detail::sub(p00, p10),
                                          detail::sub(p01, p10));
            const Vec3 n2 = detail::cross(detail::sub(p01, p10),
                                          detail::sub(p11, p10));
            if (detail::opposed(n1, n2)) {
                quad_folded[qi] = 1;
                ++c.fold_quads;
            }
            const Vec3 m1 = detail::cross(detail::sub(p01, p00),
                                          detail::sub(p11, p00));
            const Vec3 m2 = detail::cross(detail::sub(p11, p00),
                                          detail::sub(p10, p00));
            if (detail::opposed(m1, m2)) ++c.fold_quads_other_diagonal;
        }
    }
    for (size_t i = 0; i < vertex_valid.size(); ++i)
        if (vertex_valid[i] && !vertex_in_quad[i]) ++c.isolated_vertices;

    if (c.valid_quads > 0) {
        detail::DisjointSet ds((size_t)qrows * qcols);
        for (int v = 0; v < qrows; ++v)
            for (int u = 0; u < qcols; ++u) {
                const size_t qi = (size_t)v * qcols + u;
                if (!quad_valid[qi]) continue;
                if (u + 1 < qcols && quad_valid[qi + 1])
                    ds.unite((int32_t)qi, (int32_t)(qi + 1));
                if (v + 1 < qrows && quad_valid[qi + qcols])
                    ds.unite((int32_t)qi, (int32_t)(qi + qcols));
            }
        std::vector<Component> comps;
        std::vector<int32_t> root_to_comp((size_t)qrows * qcols, -1);
        for (int v = 0; v < qrows; ++v)
            for (int u = 0; u < qcols; ++u) {
                const size_t qi = (size_t)v * qcols + u;
                if (!quad_valid[qi]) continue;
                const int32_t r = ds.find((int32_t)qi);
                if (root_to_comp[r] < 0) {
                    root_to_comp[r] = (int32_t)comps.size();
                    Component nc;
                    nc.v = v;
                    nc.u = u;
                    nc.site = vec(P(v, u));
                    comps.push_back(nc);
                }
                Component& cc = comps[root_to_comp[r]];
                ++cc.quads;
                cc.area_vx2 += quad_area[qi];
            }
        c.component_count = (int64_t)comps.size();
        std::stable_sort(comps.begin(), comps.end(),
                         [](const Component& a, const Component& b) {
                             if (a.quads != b.quads) return a.quads > b.quads;
                             if (a.v != b.v) return a.v < b.v;
                             return a.u < b.u;
                         });
        for (size_t i = 1; i < comps.size(); ++i) {
            c.island_quads += comps[i].quads;
            c.island_area_vx2 += comps[i].area_vx2;
        }
        if ((int)comps.size() > params.max_sites)
            comps.resize((size_t)params.max_sites);
        c.components = std::move(comps);
    }

    {
        detail::DisjointSet ds((size_t)rows * cols);
        for (int v = 0; v < rows; ++v)
            for (int u = 0; u < cols; ++u) {
                const size_t i = (size_t)v * cols + u;
                if (vertex_valid[i]) continue;
                if (u + 1 < cols && !vertex_valid[i + 1])
                    ds.unite((int32_t)i, (int32_t)(i + 1));
                if (v + 1 < rows && !vertex_valid[i + cols])
                    ds.unite((int32_t)i, (int32_t)(i + cols));
                if (v + 1 < rows && u + 1 < cols && !vertex_valid[i + cols + 1])
                    ds.unite((int32_t)i, (int32_t)(i + cols + 1));
                if (v + 1 < rows && u > 0 && !vertex_valid[i + cols - 1])
                    ds.unite((int32_t)i, (int32_t)(i + cols - 1));
            }
        std::vector<uint8_t> touches_border((size_t)rows * cols, 0);
        auto mark = [&](int v, int u) {
            const size_t i = (size_t)v * cols + u;
            if (!vertex_valid[i]) touches_border[ds.find((int32_t)i)] = 1;
        };
        for (int u = 0; u < cols; ++u) { mark(0, u); mark(rows - 1, u); }
        for (int v = 0; v < rows; ++v) { mark(v, 0); mark(v, cols - 1); }

        std::vector<Hole> holes;
        std::vector<int32_t> root_to_hole((size_t)rows * cols, -1);
        for (int v = 0; v < rows; ++v)
            for (int u = 0; u < cols; ++u) {
                const size_t i = (size_t)v * cols + u;
                if (vertex_valid[i]) continue;
                const int32_t r = ds.find((int32_t)i);
                if (touches_border[r]) continue;
                if (root_to_hole[r] < 0) {
                    root_to_hole[r] = (int32_t)holes.size();
                    Hole nh;
                    nh.v = v;
                    nh.u = u;
                    nh.v0 = nh.v1 = v;
                    nh.u0 = nh.u1 = u;
                    holes.push_back(nh);
                }
                Hole& hh = holes[root_to_hole[r]];
                ++hh.cells;
                hh.v0 = std::min(hh.v0, v);
                hh.v1 = std::max(hh.v1, v);
                hh.u0 = std::min(hh.u0, u);
                hh.u1 = std::max(hh.u1, u);
            }
        for (Hole& h : holes) {
            c.hole_cells += h.cells;
            bool placed = false;
            for (int dv = -1; dv <= 1 && !placed; ++dv)
                for (int du = -1; du <= 1 && !placed; ++du) {
                    const int vv = h.v + dv, uu = h.u + du;
                    if (vv < 0 || uu < 0 || vv >= rows || uu >= cols) continue;
                    if (!vertex_valid[(size_t)vv * cols + uu]) continue;
                    h.site = vec(P(vv, uu));
                    placed = true;
                }
        }
        c.hole_count = (int64_t)holes.size();
        std::stable_sort(holes.begin(), holes.end(),
                         [](const Hole& a, const Hole& b) {
                             if (a.cells != b.cells) return a.cells > b.cells;
                             if (a.v != b.v) return a.v < b.v;
                             return a.u < b.u;
                         });
        if ((int)holes.size() > params.max_sites)
            holes.resize((size_t)params.max_sites);
        c.holes = std::move(holes);
    }

    {
        std::vector<float> len_u, len_v;
        len_u.reserve((size_t)rows * std::max(0, cols - 1));
        len_v.reserve((size_t)std::max(0, rows - 1) * cols);
        for (int v = 0; v < rows; ++v)
            for (int u = 0; u + 1 < cols; ++u) {
                const size_t i = (size_t)v * cols + u;
                if (!vertex_valid[i] || !vertex_valid[i + 1]) continue;
                len_u.push_back((float)dist(vec(P(v, u)), vec(P(v, u + 1))));
            }
        for (int v = 0; v + 1 < rows; ++v)
            for (int u = 0; u < cols; ++u) {
                const size_t i = (size_t)v * cols + u;
                if (!vertex_valid[i] || !vertex_valid[i + cols]) continue;
                len_v.push_back((float)dist(vec(P(v, u)), vec(P(v + 1, u))));
            }
        c.median_step_u = detail::median_of(len_u);
        c.median_step_v = detail::median_of(len_v);

        const bool have_u = c.median_step_u > 0.0;
        const bool have_v = c.median_step_v > 0.0;
        std::vector<uint8_t> torn_vertex((size_t)rows * cols, 0);
        struct TornEdge { size_t idx; float len; float ratio; };
        std::vector<TornEdge> torn;
        auto flag = [&](int v, int u, int v2, int u2, double len, double med) {
            const double ratio = len / med;
            if (ratio <= params.tear_factor) return;
            ++c.tear_edges;
            c.max_edge_ratio = std::max(c.max_edge_ratio, ratio);
            torn_vertex[(size_t)v * cols + u] = 1;
            torn_vertex[(size_t)v2 * cols + u2] = 1;
            torn.push_back({(size_t)v * cols + u, (float)len, (float)ratio});
        };
        if (have_u)
            for (int v = 0; v < rows; ++v)
                for (int u = 0; u + 1 < cols; ++u) {
                    const size_t i = (size_t)v * cols + u;
                    if (!vertex_valid[i] || !vertex_valid[i + 1]) continue;
                    flag(v, u, v, u + 1,
                         dist(vec(P(v, u)), vec(P(v, u + 1))),
                         c.median_step_u);
                }
        if (have_v)
            for (int v = 0; v + 1 < rows; ++v)
                for (int u = 0; u < cols; ++u) {
                    const size_t i = (size_t)v * cols + u;
                    if (!vertex_valid[i] || !vertex_valid[i + cols]) continue;
                    flag(v, u, v + 1, u,
                         dist(vec(P(v, u)), vec(P(v + 1, u))),
                         c.median_step_v);
                }

        if (c.tear_edges > 0) {
            detail::DisjointSet ds((size_t)rows * cols);
            for (int v = 0; v < rows; ++v)
                for (int u = 0; u < cols; ++u) {
                    const size_t i = (size_t)v * cols + u;
                    if (!torn_vertex[i]) continue;
                    if (u + 1 < cols && torn_vertex[i + 1])
                        ds.unite((int32_t)i, (int32_t)(i + 1));
                    if (v + 1 < rows && torn_vertex[i + cols])
                        ds.unite((int32_t)i, (int32_t)(i + cols));
                    if (v + 1 < rows && u + 1 < cols && torn_vertex[i + cols + 1])
                        ds.unite((int32_t)i, (int32_t)(i + cols + 1));
                    if (v + 1 < rows && u > 0 && torn_vertex[i + cols - 1])
                        ds.unite((int32_t)i, (int32_t)(i + cols - 1));
                }
            std::vector<TearSite> sites;
            std::vector<int32_t> root_to_site((size_t)rows * cols, -1);
            for (const TornEdge& e : torn) {
                const int32_t r = ds.find((int32_t)e.idx);
                if (root_to_site[r] < 0) {
                    root_to_site[r] = (int32_t)sites.size();
                    TearSite ns;
                    ns.v = (int)(e.idx / cols);
                    ns.u = (int)(e.idx % cols);
                    sites.push_back(ns);
                }
                TearSite& s = sites[root_to_site[r]];
                ++s.edges;
                s.max_ratio = std::max(s.max_ratio, (double)e.ratio);
                if ((double)e.len > s.max_len) {
                    s.max_len = (double)e.len;
                    s.v = (int)(e.idx / cols);
                    s.u = (int)(e.idx % cols);
                    s.site = vec(P(s.v, s.u));
                }
            }
            c.tear_site_count = (int64_t)sites.size();
            std::stable_sort(sites.begin(), sites.end(),
                             [](const TearSite& a, const TearSite& b) {
                                 if (a.max_len != b.max_len)
                                     return a.max_len > b.max_len;
                                 if (a.v != b.v) return a.v < b.v;
                                 return a.u < b.u;
                             });
            if ((int)sites.size() > params.max_sites)
                sites.resize((size_t)params.max_sites);
            c.tears = std::move(sites);
        }
    }

    if (c.fold_quads > 0) {
        detail::DisjointSet ds((size_t)qrows * qcols);
        for (int v = 0; v < qrows; ++v)
            for (int u = 0; u < qcols; ++u) {
                const size_t qi = (size_t)v * qcols + u;
                if (!quad_folded[qi]) continue;
                if (u + 1 < qcols && quad_folded[qi + 1])
                    ds.unite((int32_t)qi, (int32_t)(qi + 1));
                if (v + 1 < qrows && quad_folded[qi + qcols])
                    ds.unite((int32_t)qi, (int32_t)(qi + qcols));
                if (v + 1 < qrows && u + 1 < qcols && quad_folded[qi + qcols + 1])
                    ds.unite((int32_t)qi, (int32_t)(qi + qcols + 1));
                if (v + 1 < qrows && u > 0 && quad_folded[qi + qcols - 1])
                    ds.unite((int32_t)qi, (int32_t)(qi + qcols - 1));
            }
        std::vector<FoldSite> sites;
        std::vector<int32_t> root_to_site((size_t)qrows * qcols, -1);
        for (int v = 0; v < qrows; ++v)
            for (int u = 0; u < qcols; ++u) {
                const size_t qi = (size_t)v * qcols + u;
                if (!quad_folded[qi]) continue;
                const int32_t r = ds.find((int32_t)qi);
                if (root_to_site[r] < 0) {
                    root_to_site[r] = (int32_t)sites.size();
                    FoldSite ns;
                    ns.v = v;
                    ns.u = u;
                    ns.site = vec(P(v, u));
                    sites.push_back(ns);
                }
                ++sites[root_to_site[r]].quads;
            }
        c.fold_site_count = (int64_t)sites.size();
        std::stable_sort(sites.begin(), sites.end(),
                         [](const FoldSite& a, const FoldSite& b) {
                             if (a.quads != b.quads) return a.quads > b.quads;
                             if (a.v != b.v) return a.v < b.v;
                             return a.u < b.u;
                         });
        if ((int)sites.size() > params.max_sites)
            sites.resize((size_t)params.max_sites);
        c.folds = std::move(sites);
    }

    return c;
}

}  // namespace vc_topology
