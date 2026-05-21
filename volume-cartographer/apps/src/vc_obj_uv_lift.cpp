// vc_obj_uv_lift.cpp
//
// Lift per-vertex UVs from a flattened coarse mesh onto a high-resolution
// mesh of the same tifxyz surface. Both meshes are assumed to have come from
// vc_tifxyz2obj on the same source tifxyz grid (the fine without decimation
// or at lower decimation level, the coarse with higher decimation), so their
// OBJ texcoords encode the original source-grid index times a constant
// (1/meta.scale). That makes the lift exact in 2D grid space: each fine
// vertex sits inside exactly one quad of the coarse grid, no spatial /
// nearest-triangle search needed. This avoids the streak / tear artefacts
// you get from a 3D-nearest lift on multi-layer surfaces (a scroll segment
// can pass close to itself in 3D while being far apart on the surface).
//
// Inputs:
//   coarse_grid.obj  fine mesh's coarse partner from vc_tifxyz2obj.
//                    UVs = source-grid index * (1/scale). Provides the
//                    grid-space layout of the coarse triangulation.
//   coarse_flat.obj  same triangulation as coarse_grid, but UVs replaced by
//                    the flattened (SLIM) UVs. Vertex ordering must match
//                    coarse_grid.obj (this is the case as long as both came
//                    out of the same vc_tifxyz2obj run -> flatboi).
//   fine_grid.obj    full-res mesh from vc_tifxyz2obj. UVs = source-grid
//                    index * (1/scale), same convention as coarse_grid.
//
// Output:
//   out.obj          fine mesh with per-corner UVs lifted from coarse_flat.

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>

#include <Eigen/Core>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;

static void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " <coarse_grid.obj> <coarse_flat.obj> <fine_grid.obj> <out.obj>\n"
              << "  coarse_grid.obj  vc_tifxyz2obj output of the coarse mesh (grid UVs)\n"
              << "  coarse_flat.obj  flatboi output of coarse_grid.obj (flat UVs)\n"
              << "  fine_grid.obj    vc_tifxyz2obj output of the fine mesh (grid UVs)\n"
              << "  out.obj          fine mesh with lifted flat UVs\n";
}

int main(int argc, char** argv) {
    if (argc != 5) { print_usage(argv[0]); return 1; }

    const fs::path coarse_grid_path = argv[1];
    const fs::path coarse_flat_path = argv[2];
    const fs::path fine_grid_path   = argv[3];
    const fs::path out_path         = argv[4];

    for (const auto& p : {coarse_grid_path, coarse_flat_path, fine_grid_path}) {
        if (!fs::exists(p)) { std::cerr << "not found: " << p << "\n"; return 1; }
    }

    Eigen::MatrixXd Vcg, TCcg, Ncg;
    Eigen::MatrixXi Fcg, FTCcg, FNcg;
    if (!igl::readOBJ(coarse_grid_path.string(), Vcg, TCcg, Ncg, Fcg, FTCcg, FNcg)) {
        std::cerr << "failed to read " << coarse_grid_path << "\n"; return 2;
    }

    Eigen::MatrixXd Vcf, TCcf, Ncf;
    Eigen::MatrixXi Fcf, FTCcf, FNcf;
    if (!igl::readOBJ(coarse_flat_path.string(), Vcf, TCcf, Ncf, Fcf, FTCcf, FNcf)) {
        std::cerr << "failed to read " << coarse_flat_path << "\n"; return 2;
    }
    if (TCcf.rows() == 0 || FTCcf.rows() == 0) {
        std::cerr << "coarse_flat.obj has no UVs\n"; return 2;
    }
    if (!TCcf.allFinite()) {
        std::cerr << "coarse_flat.obj has non-finite UVs (NaN/Inf). "
                  << "Upstream flatten diverged; refusing to propagate.\n";
        return 4;
    }
    if (Vcg.rows() != Vcf.rows() || Fcg.rows() != Fcf.rows()) {
        std::cerr << "coarse_grid and coarse_flat must share vertex/face counts "
                  << "(V " << Vcg.rows() << " vs " << Vcf.rows()
                  << ", F " << Fcg.rows() << " vs " << Fcf.rows() << ")\n";
        return 2;
    }
    if (TCcg.rows() == 0 || FTCcg.rows() == 0) {
        std::cerr << "coarse_grid.obj has no UVs (need grid-space UVs from vc_tifxyz2obj)\n";
        return 2;
    }

    Eigen::MatrixXd Vf, TCf, Nf;
    Eigen::MatrixXi Ff, FTCf, FNf;
    if (!igl::readOBJ(fine_grid_path.string(), Vf, TCf, Nf, Ff, FTCf, FNf)) {
        std::cerr << "failed to read " << fine_grid_path << "\n"; return 2;
    }
    if (TCf.rows() == 0 || FTCf.rows() == 0) {
        std::cerr << "fine_grid.obj has no UVs (need grid-space UVs from vc_tifxyz2obj)\n";
        return 2;
    }

    std::cout << "Coarse grid: V=" << Vcg.rows() << " F=" << Fcg.rows()
              << " TC=" << TCcg.rows() << "\n";
    std::cout << "Coarse flat: V=" << Vcf.rows() << " F=" << Fcf.rows()
              << " TC=" << TCcf.rows() << "\n";
    std::cout << "Fine grid:   V=" << Vf.rows()  << " F=" << Ff.rows()
              << " TC=" << TCf.rows() << "\n";

    // Per-coarse-vertex grid-UV: average the per-corner grid UV across all
    // corners that reference this vertex. (vc_tifxyz2obj actually assigns one
    // (vt) per unique grid cell, so all corners for a given vertex share the
    // same UV; the loop is just defensive.)
    Eigen::MatrixXd vert_uv = Eigen::MatrixXd::Zero(Vcg.rows(), 2);
    Eigen::VectorXi vert_uv_n = Eigen::VectorXi::Zero(Vcg.rows());
    for (int t = 0; t < Fcg.rows(); ++t) {
        for (int k = 0; k < 3; ++k) {
            const int v  = Fcg(t, k);
            const int tc = FTCcg(t, k);
            vert_uv.row(v) += TCcg.row(tc);
            vert_uv_n(v)   += 1;
        }
    }
    for (int i = 0; i < Vcg.rows(); ++i) {
        if (vert_uv_n(i) > 0) vert_uv.row(i) /= double(vert_uv_n(i));
    }

    // Per-fine-vertex grid UV (averaged across corners as above).
    // Computed early so we can derive the coarse->fine UV scale from bboxes.
    Eigen::MatrixXd fine_grid_uv = Eigen::MatrixXd::Zero(Vf.rows(), 2);
    Eigen::VectorXi fine_grid_n  = Eigen::VectorXi::Zero(Vf.rows());
    for (int t = 0; t < Ff.rows(); ++t) {
        for (int k = 0; k < 3; ++k) {
            fine_grid_uv.row(Ff(t, k)) += TCf.row(FTCf(t, k));
            fine_grid_n(Ff(t, k))      += 1;
        }
    }
    for (int i = 0; i < Vf.rows(); ++i) {
        if (fine_grid_n(i) > 0) fine_grid_uv.row(i) /= double(fine_grid_n(i));
    }

    // The coarse and fine OBJs both encode source-grid-pixel * (1/scale) as
    // their UVs, BUT the coarse mesh's "source grid" is itself the decimated
    // grid. So coarse_UV * stride^N == fine_UV. Recover that scale from the
    // bbox ratios so we don't need to know the decimation level here. Apply
    // to the coarse-grid UVs so both meshes live in the same coordinate
    // system.
    {
        Eigen::RowVector2d cmn = vert_uv.colwise().minCoeff();
        Eigen::RowVector2d cmx = vert_uv.colwise().maxCoeff();
        Eigen::RowVector2d fmn = fine_grid_uv.colwise().minCoeff();
        Eigen::RowVector2d fmx = fine_grid_uv.colwise().maxCoeff();
        const double cu_span = std::max(1e-12, cmx(0) - cmn(0));
        const double cv_span = std::max(1e-12, cmx(1) - cmn(1));
        const double fu_span = std::max(1e-12, fmx(0) - fmn(0));
        const double fv_span = std::max(1e-12, fmx(1) - fmn(1));
        const double su = fu_span / cu_span;
        const double sv = fv_span / cv_span;
        std::cout << "Coarse->fine UV scale: u=" << su << " v=" << sv
                  << "  (coarse bbox " << cu_span << "x" << cv_span
                  << ", fine bbox " << fu_span << "x" << fv_span << ")\n";
        // Rebase coarse UVs onto fine UV origin and rescale to fine units.
        for (int i = 0; i < vert_uv.rows(); ++i) {
            vert_uv(i, 0) = fmn(0) + (vert_uv(i, 0) - cmn(0)) * su;
            vert_uv(i, 1) = fmn(1) + (vert_uv(i, 1) - cmn(1)) * sv;
        }
    }

    // AABB of all coarse grid-UVs (now in fine-UV coords) and bin sizing.
    Eigen::RowVector2d uv_min = vert_uv.colwise().minCoeff();
    Eigen::RowVector2d uv_max = vert_uv.colwise().maxCoeff();
    const double udu = std::max(1e-12, uv_max(0) - uv_min(0));
    const double vdu = std::max(1e-12, uv_max(1) - uv_min(1));

    // Estimate cell size in coarse-UV from coarse vertex count: ~sqrt(V/area).
    const double cells_per_axis = std::max(8.0, std::sqrt(double(Vcg.rows())) * 0.5);
    const double cu = udu / cells_per_axis;
    const double cv = vdu / cells_per_axis;
    const int    bins_u = std::max(1, int(std::ceil(udu / cu)));
    const int    bins_v = std::max(1, int(std::ceil(vdu / cv)));
    std::cout << "Lift hash grid: " << bins_u << " x " << bins_v
              << " (coarse-UV bbox [" << uv_min(0) << "," << uv_min(1)
              << "] -> [" << uv_max(0) << "," << uv_max(1) << "])\n";

    auto bin_idx = [&](double u, double v) -> std::pair<int,int> {
        int bu = int(std::floor((u - uv_min(0)) / cu));
        int bv = int(std::floor((v - uv_min(1)) / cv));
        bu = std::clamp(bu, 0, bins_u - 1);
        bv = std::clamp(bv, 0, bins_v - 1);
        return {bu, bv};
    };

    std::vector<std::vector<int>> bins(static_cast<size_t>(bins_u) * bins_v);
    auto bin_at = [&](int bu, int bv) -> std::vector<int>& {
        return bins[size_t(bv) * bins_u + bu];
    };

    // For each coarse triangle, push its index into every bin its UV-bbox
    // overlaps.
    for (int t = 0; t < Fcg.rows(); ++t) {
        const Eigen::RowVector2d a = vert_uv.row(Fcg(t, 0));
        const Eigen::RowVector2d b = vert_uv.row(Fcg(t, 1));
        const Eigen::RowVector2d c = vert_uv.row(Fcg(t, 2));
        const double tu_min = std::min({a(0), b(0), c(0)});
        const double tu_max = std::max({a(0), b(0), c(0)});
        const double tv_min = std::min({a(1), b(1), c(1)});
        const double tv_max = std::max({a(1), b(1), c(1)});
        auto [bu0, bv0] = bin_idx(tu_min, tv_min);
        auto [bu1, bv1] = bin_idx(tu_max, tv_max);
        for (int bv = bv0; bv <= bv1; ++bv)
            for (int bu = bu0; bu <= bu1; ++bu)
                bin_at(bu, bv).push_back(t);
    }

    auto point_in_tri = [](const Eigen::RowVector2d& p,
                           const Eigen::RowVector2d& a,
                           const Eigen::RowVector2d& b,
                           const Eigen::RowVector2d& c,
                           double& w0, double& w1, double& w2) -> bool {
        const double v0u = b(0) - a(0), v0v = b(1) - a(1);
        const double v1u = c(0) - a(0), v1v = c(1) - a(1);
        const double v2u = p(0) - a(0), v2v = p(1) - a(1);
        const double den = v0u * v1v - v1u * v0v;
        if (std::abs(den) < 1e-30) return false;
        w1 = (v2u * v1v - v1u * v2v) / den;
        w2 = (v0u * v2v - v2u * v0v) / den;
        w0 = 1.0 - w1 - w2;
        const double eps = -1e-9;
        return (w0 >= eps && w1 >= eps && w2 >= eps);
    };

    // Per-coarse-vertex flat UV (same averaging trick).
    Eigen::MatrixXd flat_vert_uv = Eigen::MatrixXd::Zero(Vcf.rows(), 2);
    Eigen::VectorXi flat_vert_n  = Eigen::VectorXi::Zero(Vcf.rows());
    for (int t = 0; t < Fcf.rows(); ++t) {
        for (int k = 0; k < 3; ++k) {
            flat_vert_uv.row(Fcf(t, k)) += TCcf.row(FTCcf(t, k));
            flat_vert_n(Fcf(t, k))      += 1;
        }
    }
    for (int i = 0; i < Vcf.rows(); ++i) {
        if (flat_vert_n(i) > 0) flat_vert_uv.row(i) /= double(flat_vert_n(i));
    }

    Eigen::MatrixXd UVf(Vf.rows(), 2);
    std::atomic<long> misses{0};
    std::atomic<long> fallback{0};
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < Vf.rows(); ++i) {
        const Eigen::RowVector2d p = fine_grid_uv.row(i);
        auto [bu, bv] = bin_idx(p(0), p(1));
        bool found = false;
        double bw0 = 0, bw1 = 0, bw2 = 0;
        int bt = -1;
        // Look in the bin and its neighbors. We widen the radius for the
        // border row/column where a fine vertex's nominal grid-UV can sit
        // slightly outside the convex hull of any coarse triangle (e.g. the
        // last fine row that decimation dropped).
        for (int dv = -2; dv <= 2 && !found; ++dv) {
            for (int du = -2; du <= 2 && !found; ++du) {
                const int qu = bu + du, qv = bv + dv;
                if (qu < 0 || qv < 0 || qu >= bins_u || qv >= bins_v) continue;
                for (int t : bin_at(qu, qv)) {
                    double w0, w1, w2;
                    const Eigen::RowVector2d a = vert_uv.row(Fcg(t, 0));
                    const Eigen::RowVector2d b = vert_uv.row(Fcg(t, 1));
                    const Eigen::RowVector2d c = vert_uv.row(Fcg(t, 2));
                    if (point_in_tri(p, a, b, c, w0, w1, w2)) {
                        bw0 = w0; bw1 = w1; bw2 = w2;
                        bt = t;
                        found = true;
                        break;
                    }
                }
            }
        }
        if (!found) {
            // Fallback for fine vertices that sit just outside the coarse
            // triangulation (boundary row/col dropped by decimation): clamp p
            // into the coarse-UV bbox and re-search a small neighborhood.
            // Avoids the O(V_coarse) brute-force loop that would otherwise
            // dominate runtime on big meshes.
            misses.fetch_add(1, std::memory_order_relaxed);
            Eigen::RowVector2d pc(
                std::clamp(p(0), uv_min(0), uv_max(0)),
                std::clamp(p(1), uv_min(1), uv_max(1)));
            auto [bu2, bv2] = bin_idx(pc(0), pc(1));
            for (int dv = -2; dv <= 2 && !found; ++dv) {
                for (int du = -2; du <= 2 && !found; ++du) {
                    const int qu = bu2 + du, qv = bv2 + dv;
                    if (qu < 0 || qv < 0 || qu >= bins_u || qv >= bins_v) continue;
                    for (int t : bin_at(qu, qv)) {
                        double w0, w1, w2;
                        const Eigen::RowVector2d a = vert_uv.row(Fcg(t, 0));
                        const Eigen::RowVector2d b = vert_uv.row(Fcg(t, 1));
                        const Eigen::RowVector2d c = vert_uv.row(Fcg(t, 2));
                        if (point_in_tri(pc, a, b, c, w0, w1, w2)) {
                            bw0 = w0; bw1 = w1; bw2 = w2;
                            bt = t;
                            found = true;
                            break;
                        }
                    }
                }
            }
            if (!found) {
                // Last resort: nearest coarse vertex in this bin / neighbors.
                int best = -1;
                double bestD = std::numeric_limits<double>::infinity();
                for (int dv = -2; dv <= 2; ++dv) {
                    for (int du = -2; du <= 2; ++du) {
                        const int qu = bu2 + du, qv = bv2 + dv;
                        if (qu < 0 || qv < 0 || qu >= bins_u || qv >= bins_v) continue;
                        for (int t : bin_at(qu, qv)) {
                            for (int k = 0; k < 3; ++k) {
                                const int v = Fcg(t, k);
                                const double du_ = vert_uv(v, 0) - pc(0);
                                const double dv_ = vert_uv(v, 1) - pc(1);
                                const double d = du_*du_ + dv_*dv_;
                                if (d < bestD) { bestD = d; best = v; }
                            }
                        }
                    }
                }
                if (best >= 0) {
                    UVf.row(i) = flat_vert_uv.row(best);
                    fallback.fetch_add(1, std::memory_order_relaxed);
                } else {
                    UVf.row(i).setZero();
                }
                continue;
            }
            // Fell through here means clamp+expand found a triangle -> fall
            // through to the normal barycentric write below.
        }
        UVf.row(i) = bw0 * flat_vert_uv.row(Fcg(bt, 0))
                   + bw1 * flat_vert_uv.row(Fcg(bt, 1))
                   + bw2 * flat_vert_uv.row(Fcg(bt, 2));
    }
    if (misses.load() > 0) {
        std::cout << "Lift fell back to nearest-vertex for " << misses.load()
                  << " / " << Vf.rows() << " fine vertices"
                  << " (resolved=" << fallback.load() << ")\n";
    }

    Eigen::MatrixXd UVc_out(Ff.rows() * 3, 2);
    Eigen::MatrixXi FUV(Ff.rows(), 3);
    for (int t = 0; t < Ff.rows(); ++t) {
        for (int k = 0; k < 3; ++k) {
            const int corner = t * 3 + k;
            UVc_out.row(corner) = UVf.row(Ff(t, k));
            FUV(t, k) = corner;
        }
    }

    Eigen::MatrixXd CN; CN.resize(0, 3);
    Eigen::MatrixXi FN; FN.resize(0, 3);
    if (!igl::writeOBJ(out_path.string(), Vf, Ff, CN, FN, UVc_out, FUV)) {
        std::cerr << "failed to write " << out_path << "\n"; return 3;
    }
    std::cout << "Wrote: " << out_path << "\n";
    return 0;
}
