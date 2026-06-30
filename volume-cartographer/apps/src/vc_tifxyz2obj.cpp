// vc_tifxyz2obj.cpp
#include <opencv2/core.hpp>  // operator<<(ostream, cv::Size) for the debug prints below

#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/InpaintSurface.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include "vc/core/types/VcDataset.hpp"
#include "utils/Json.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <limits>


using Json = utils::Json;

// ---- small helpers ---------------------------------------------------------
static inline bool finite3(const cv::Vec3f& v) {
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}
static inline cv::Vec3f unit_or_default(const cv::Vec3f& n, const cv::Vec3f& def={0.f,0.f,1.f}) {
    float L2 = n.dot(n);
    if (!std::isfinite(L2) || L2 <= 1e-24f) return def;
    cv::Vec3f u = n / std::sqrt(L2);
    return finite3(u) ? u : def;
}
static inline cv::Vec3f fd_fallback_normal(const cv::Mat_<cv::Vec3f>& P, int y, int x) {
    // Use local finite differences (very cheap) as a robust fallback
    cv::Vec3f dx(0,0,0), dy(0,0,0);
    bool okx=false, oky=false;

    if (x+1 < P.cols && P(y,x+1)[0] != -1) { dx = P(y,x+1) - P(y,x); okx=true; }
    else if (x-1 >= 0 && P(y,x-1)[0] != -1) { dx = P(y,x) - P(y,x-1); okx=true; }

    if (y+1 < P.rows && P(y+1,x)[0] != -1) { dy = P(y+1,x) - P(y,x); oky=true; }
    else if (y-1 >= 0 && P(y-1,x)[0] != -1) { dy = P(y,x) - P(y-1,x); oky=true; }

    if (okx && oky) return unit_or_default(dx.cross(dy));
    return {0.f,0.f,1.f};
}
// ---------------------------------------------------------------------------

// Decimates a grid by a target ratio (e.g., 0.5 keeps ~50% of points)
// Computes the appropriate stride from the ratio: stride = 1/sqrt(ratio)
static cv::Mat_<cv::Vec3f> decimate_grid_ratio(const cv::Mat_<cv::Vec3f>& points, float ratio)
{
    if (ratio <= 0.0f || ratio >= 1.0f) return points.clone();

    // For ratio r, we want 1/s² = r, so s = 1/sqrt(r)
    int stride = std::max(2, static_cast<int>(std::round(1.0f / std::sqrt(ratio))));

    // Calculate new dimensions
    int new_rows = (points.rows + stride - 1) / stride;
    int new_cols = (points.cols + stride - 1) / stride;

    cv::Mat_<cv::Vec3f> decimated(new_rows, new_cols);

    // Sample every stride-th point
    for (int j = 0; j < new_rows; ++j) {
        for (int i = 0; i < new_cols; ++i) {
            int src_j = j * stride;
            int src_i = i * stride;

            // Ensure we don't go out of bounds
            if (src_j < points.rows && src_i < points.cols) {
                decimated(j, i) = points(src_j, src_i);
            } else {
                // Handle edge case - use the last valid point
                src_j = std::min(src_j, points.rows - 1);
                src_i = std::min(src_i, points.cols - 1);
                decimated(j, i) = points(src_j, src_i);
            }
        }
    }

    float actual_ratio = float(new_rows * new_cols) / float(points.rows * points.cols);
    std::cout << "Decimation with ratio " << ratio << " (stride=" << stride << "): "
              << "reduced to " << new_rows << " x " << new_cols
              << " (" << (new_rows * new_cols) << " points, actual ratio: "
              << std::fixed << std::setprecision(3) << actual_ratio << ")" << std::endl;

    return decimated;
}



// ---------------------------------------------------------------------------

// Adds vertex/texcoord/normal for grid location (y,x) if not already added.
// Returns the (1-based) OBJ index for this vertex (and matching vt/vn).
static int get_add_vertex(std::ofstream& out,
                          std::ofstream* griduv,
                          const cv::Mat_<cv::Vec3f>& points,
                          const cv::Mat_<cv::Vec3f>& normals,
                          cv::Mat_<int>& idxs,
                          int& v_idx,
                          cv::Vec2i loc,
                          bool normalize_uv,
                          float uv_fac_x,
                          float uv_fac_y)
{

    if (idxs(loc) == -1) {
        idxs(loc) = v_idx++;
        const cv::Vec3f p = points(loc);
        out << "v " << p[0] << " " << p[1] << " " << p[2] << '\n';

        // Grid-UV sidecar: raw (col,row) of this vertex in source-grid space,
        // one line per OBJ vertex in emit order. Carries the original grid
        // cell through flatten so approval can be resampled. See vc_obj2tifxyz.
        if (griduv) {
            (*griduv) << loc[1] << ' ' << loc[0] << '\n';
        }

        // UVs: scaled by SCALE = 20
        const float u = normalize_uv ? float(loc[1]) / float(points.cols - 1) : float(loc[1]) * uv_fac_x;
        const float v = normalize_uv ? float(loc[0]) / float(points.rows - 1) : float(loc[0]) * uv_fac_y;
        out << "vt " << u << " " << v << '\n';

        // Prefer precomputed per-vertex normal; validate; then fallback.
        cv::Vec3f n = normals(loc);
        bool ok = finite3(n);
        if (ok) n = unit_or_default(n);

        if (!ok || (n[0] == 0.f && n[1] == 0.f && n[2] == 0.f)) {
            // fall back to grid_normal (expects x,y), then to finite-diff
            cv::Vec3f ng = grid_normal(points, cv::Vec3f(float(loc[1]), float(loc[0]), 0.f));
            if (finite3(ng)) n = unit_or_default(ng);
            else             n = fd_fallback_normal(points, loc[0], loc[1]);
        }
        out << "vn " << n[0] << " " << n[1] << " " << n[2] << '\n';
    }

    return idxs(loc);
}

static cv::Mat_<cv::Vec3f> build_vertex_normals_from_faces(
        const cv::Mat_<cv::Vec3f>& P)
{
    cv::Mat_<cv::Vec3f> nsum(P.size(), cv::Vec3f(0,0,0));
    cv::Mat_<int>       ncnt(P.size(), 0);

    for (int j = 0; j < P.rows - 1; ++j) {
        for (int i = 0; i < P.cols - 1; ++i) {
            if (!loc_valid(P, cv::Vec2d(j, i))) continue;

            const cv::Vec3f p00 = P(j,   i  );
            const cv::Vec3f p01 = P(j,   i+1);
            const cv::Vec3f p10 = P(j+1, i  );
            const cv::Vec3f p11 = P(j+1, i+1);

            // Face winding matches your 'f' lines:
            // f c10 c00 c01   and   f c10 c01 c11
            const cv::Vec3f n1 = (p00 - p10).cross(p01 - p10); // (c10,c00,c01)
            const cv::Vec3f n2 = (p01 - p10).cross(p11 - p10); // (c10,c01,c11)

            auto add = [&](int y, int x, const cv::Vec3f& n) {
                nsum(y,x) += n;
                ncnt(y,x) += 1;
            };
            add(j+1,i, n1); add(j,i, n1); add(j,i+1, n1);
            add(j+1,i, n2); add(j,i+1, n2); add(j+1,i+1, n2);
        }
    }

    // normalize (and guard against degenerate cases)
    for (int y = 0; y < P.rows; ++y)
        for (int x = 0; x < P.cols; ++x) {
            cv::Vec3f n = nsum(y,x);
            float L2 = n.dot(n);
            if (ncnt(y,x) > 0 && std::isfinite(L2) && L2 > 1e-20f)
                nsum(y,x) = n / std::sqrt(L2);
            else
                nsum(y,x) = cv::Vec3f(0,0,1); // safe default (rarely used now)
        }

    return nsum;
}

static void surf_write_obj(QuadSurface *surf, const std::filesystem::path &out_fn, bool normalize_uv, bool align_grid, float keep_percent, bool clean_surface, float clean_sigma_k, bool inpaint_holes)
{
    cv::Mat_<cv::Vec3f> points = surf->rawPoints();

    // Heal degenerate (collapsed grid-adjacent) cells before anything else.
    // These tracer defects make zero-area triangles that NaN SLIM at iter 1;
    // healthy adjacent cells are ~1/scale voxels apart. No-op if none found.
    {
        const cv::Vec2f sc = surf->scale();
        const double scale = (std::isfinite(sc[0]) && sc[0] > 0.f) ? sc[0] : 0.0;
        const int healed = vc::core::util::healDegenerateCells(points, scale);
        if (healed > 0) {
            std::cerr << "note: healed " << healed
                      << " degenerate (collapsed) grid cell(s) before flattening\n";
        }
    }

    // Clean surface outliers if requested
    if (clean_surface) {
        std::cout << "Cleaning surface outliers..." << std::endl;
        points = clean_surface_outliers(points, clean_sigma_k);
    }
    
    // If align_grid is enabled, align Z only:
    // - Rows: flatten Z per row (keep original X and Y)
    if (align_grid) {
        // Precompute row averages for Z only
        std::vector<float> row_avg_z(points.rows, std::numeric_limits<float>::quiet_NaN());

        // Row averages (Z only)
        for (int j = 0; j < points.rows; ++j) {
            double z_sum = 0.0; int n = 0;
            for (int i = 0; i < points.cols; ++i) {
                if (points(j, i)[0] != -1) { // valid point: x != -1 sentinel
                    z_sum += points(j, i)[2];
                    ++n;
                }
            }
            if (n > 0) {
                row_avg_z[j] = static_cast<float>(z_sum / n);
            }
        }

        // Apply alignment
        for (int j = 0; j < points.rows; ++j) {
            for (int i = 0; i < points.cols; ++i) {
                if (points(j, i)[0] == -1) continue;
                if (std::isfinite(row_avg_z[j])) points(j, i)[2] = row_avg_z[j];
            }
        }
    }
    
    // Apply decimation if requested: one pass, stride derived from the
    // requested keep fraction (decimate_grid_ratio computes stride =
    // round(1/sqrt(p))). Skip when keep_percent >= 100 (no decimation).
    if (keep_percent > 0.0f && keep_percent < 100.0f) {
        std::cout << "Original grid: " << points.rows << " x " << points.cols
                  << " (" << (points.rows * points.cols) << " points)" << std::endl;
        points = decimate_grid_ratio(points, keep_percent / 100.0f);
    }

    // Fill isolated interior holes that would otherwise create extra boundary
    // loops in the OBJ and break downstream flattening (SLIM goes NaN).
    if (inpaint_holes) {
        const int filled = vc::core::util::inpaintSurfaceHoles(points);
        if (filled > 0) {
            std::cerr << "warning: inpainted " << filled
                      << " invalid surface cell(s) from local neighbors "
                      << "(holes can break flattening; pass --no-inpaint to disable)\n";
        }
    }

    cv::Mat_<int> idxs(points.size(), -1);

    std::ofstream out(out_fn);
    if (!out) {
        std::cerr << "Failed to open for write: " << out_fn << "\n";
        return;
    }
    out << std::fixed << std::setprecision(6);

    // Emit a grid-UV sidecar only when the source carries an approval mask;
    // it lets the flatten pipeline resample approval onto the new grid.
    std::ofstream griduv;
    std::ofstream* griduv_ptr = nullptr;
    if (!surf->channel("approval", SURF_CHANNEL_NORESIZE).empty()) {
        const std::filesystem::path griduv_fn = out_fn.string() + ".griduv";
        griduv.open(griduv_fn);
        if (griduv) {
            griduv_ptr = &griduv;
        } else {
            std::cerr << "warning: could not write grid-UV sidecar: " << griduv_fn << "\n";
        }
    }

    cv::Mat_<cv::Vec3f> normals = build_vertex_normals_from_faces(points);

    std::cout << "Point dims: " << points.size()
              << " cols: " << points.cols
              << " rows: " << points.rows << std::endl;
    
    if (align_grid) {
        std::cout << "Grid alignment: enabled (rows: constant Z only)\n";
    }

    // Derive UV scale from meta: surf->scale() is typically micrometers-per-pixel (or similar).
    // You asked to use the reciprocal (1/scale) as the multiplier.
    cv::Vec2f s = surf->scale();           // [sx, sy]
    float uv_fac_x = (std::isfinite(s[0]) && s[0] > 0.f) ? 1.0f / s[0] : 1.0f;
    float uv_fac_y = (std::isfinite(s[1]) && s[1] > 0.f) ? 1.0f / s[1] : 1.0f;
    if (normalize_uv) {
        std::cout << "UVs: normalized to [0,1]\n";
    } else {
        std::cout << "UVs: scaled by 1/scale from meta.json  (u*= " << uv_fac_x
                  << ", v*= " << uv_fac_y << " )\n";
        std::cout << "      (meta scale = [" << s[0] << ", " << s[1] << "])\n";
    }

    // Expected healthy adjacent-cell spacing in voxels (= 1/scale); used to
    // judge degenerate triangles scale-relatively below.
    const double expected_step = (std::isfinite(uv_fac_x) && uv_fac_x > 0.f) ? double(uv_fac_x) : 1.0;

    // Guard: never emit a zero-area triangle. healDegenerateCells repairs the
    // vast majority of collapsed cells, but a few (e.g. fully-collapsed 2x2
    // blocks) can't be separated locally -- and even one zero-area triangle
    // makes SLIM's symmetric-Dirichlet system singular and NaN at iter 1. Skip
    // any triangle whose three grid corners aren't geometrically distinct; the
    // worst case is a sub-cell hole, which the flattener tolerates.
    auto tri_ok = [&](cv::Vec2i A, cv::Vec2i B, cv::Vec2i C) -> bool {
        const cv::Vec3d pa = points(A[0], A[1]);
        const cv::Vec3d pb = points(B[0], B[1]);
        const cv::Vec3d pc = points(C[0], C[1]);
        const cv::Vec3d ab = pb - pa, ac = pc - pa;
        // Reject if any edge is (near-)zero, or the triangle is a sliver:
        // 2*area / (longest edge) is the shortest height; require it to be a
        // meaningful fraction of the expected cell spacing. Scale-relative so
        // it works regardless of coordinate magnitude (~1e4 here).
        const double bc = cv::norm(pc - pb);
        const double eAB = cv::norm(ab), eAC = cv::norm(ac);
        const double emin = std::min(std::min(eAB, eAC), bc);
        const double emax = std::max(std::max(eAB, eAC), bc);
        if (emin < 1e-3 * expected_step) return false;     // collapsed edge
        const double twiceArea = cv::norm(ab.cross(ac));
        if (emax <= 0.0) return false;
        const double height = twiceArea / emax;            // shortest altitude
        return height > 1e-3 * expected_step;              // not a sliver
    };

    // Two passes so we never emit an orphan vertex (one referenced by no
    // surviving triangle). An orphan leaves an all-zero row/col in SLIM's
    // L = AᵀWA, which crashes the PaStiX Cholesky (heap corruption / SIGSEGV
    // during factorization). Pass 1: collect surviving triangles as grid-cell
    // triples and mark which cells are actually used. Pass 2: emit only used
    // vertices (get_add_vertex dedups via idxs), then the faces.
    struct Tri { cv::Vec2i a, b, c; };
    std::vector<Tri> tris;
    tris.reserve(static_cast<size_t>(points.rows) * points.cols * 2);
    cv::Mat_<uchar> used(points.size(), uchar(0));

    int n_skipped = 0;
    for (int j = 0; j < points.rows - 1; ++j)
        for (int i = 0; i < points.cols - 1; ++i)
            if (loc_valid(points, cv::Vec2d(j, i)))
            {
                const cv::Vec2i L00{j, i}, L01{j, i+1}, L10{j+1, i}, L11{j+1, i+1};
                if (tri_ok(L10, L00, L01)) {
                    tris.push_back({L10, L00, L01});
                    used(L10) = used(L00) = used(L01) = 1;
                } else { ++n_skipped; }
                if (tri_ok(L10, L01, L11)) {
                    tris.push_back({L10, L01, L11});
                    used(L10) = used(L01) = used(L11) = 1;
                } else { ++n_skipped; }
            }

    int v_idx = 1;
    auto idx_of = [&](const cv::Vec2i& L) {
        return get_add_vertex(out, griduv_ptr, points, normals, idxs, v_idx, L,
                              normalize_uv, uv_fac_x, uv_fac_y);
    };
    for (const Tri& t : tris) {
        const int a = idx_of(t.a), b = idx_of(t.b), c = idx_of(t.c);
        // same index for v/vt/vn
        out << "f " << a << "/" << a << "/" << a << " "
                   << b << "/" << b << "/" << b << " "
                   << c << "/" << c << "/" << c << '\n';
    }
    if (n_skipped > 0) {
        std::cerr << "note: skipped " << n_skipped
                  << " degenerate (zero-area) triangle(s) at emit\n";
    }
}

int main(int argc, char *argv[])
{
    if (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        std::cout << "usage: " << argv[0] << " <tiffxyz> <obj> [--normalize-uv] [--align-grid] [--keep=<p>] [--clean [K]] [--no-inpaint]\n"
                  << "  --normalize-uv : Normalize UVs to [0,1] range\n"
                  << "  --align-grid   : Align grid Z only (flatten Z per row)\n"
                  << "  --keep=<p>     : Percent of source points to keep (1..100). 100 = no decimation. One pass; stride = round(1/sqrt(p/100)).\n"
                  << "  --clean [K]    : Remove outlier points far from surface using robust distance threshold; K is sigma multiplier (default 5.0)\n"
                  << "  --inpaint      : Fill isolated invalid cells via Ceres-smoothness solve before emitting OBJ. Off by default.\n"
                  << "  --no-inpaint   : Legacy alias (inpaint is off by default now).\n";
        return EXIT_SUCCESS;
    }

    if (argc < 3) {
    std::cerr << "error: too few arguments\n"
                  << "usage: " << argv[0] << " <tiffxyz> <obj> [--normalize-uv] [--align-grid] [--keep=<p>] [--clean [K]] [--no-inpaint]\n";
        return EXIT_FAILURE;
    }

    bool normalize_uv = false;
    bool align_grid = false;
    bool clean_surface = false;
    float clean_sigma_k = 5.0f; // default K
    float keep_percent = 100.0f; // 100 = no decimation
    bool inpaint_holes = false;
    
    // Parse optional arguments
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--normalize-uv") {
            normalize_uv = true;
        } else if (arg == "--align-grid") {
            align_grid = true;
        } else if (arg.rfind("--keep=", 0) == 0) {
            try {
                float p = std::stof(arg.substr(7));
                if (p > 0.0f && p <= 100.0f) keep_percent = p;
                else {
                    std::cerr << "error: --keep must be in (0, 100]\n";
                    return EXIT_FAILURE;
                }
            } catch (...) {
                std::cerr << "error: --keep needs a numeric percent\n";
                return EXIT_FAILURE;
            }
        } else if (arg == "--inpaint") {
            inpaint_holes = true;
        } else if (arg == "--no-inpaint") {
            inpaint_holes = false;
        } else if (arg == "--clean") {
            clean_surface = true;
            // Optional numeric K argument following --clean
            if (i + 1 < argc) {
                std::string next = argv[i + 1];
                try {
                    // Allow floats (e.g., 3, 3.5)
                    float k = std::stof(next);
                    if (std::isfinite(k) && k >= 0.0f) {
                        clean_sigma_k = k;
                        ++i; // consume K
                    }
                } catch (...) {
                    // Next token is not a number; keep default K
                }
            }
        } else {
            std::cerr << "error: unknown option '" << arg << "'\n";
            return EXIT_FAILURE;
        }
    }

    const std::filesystem::path seg_path = argv[1];
    const std::filesystem::path obj_path = argv[2];

    std::unique_ptr<QuadSurface> surf;
    try {
        surf = load_quad_from_tifxyz(seg_path);
    }
    catch (...) {
        std::cerr << "error when loading: " << seg_path << std::endl;
        return EXIT_FAILURE;
    }

    surf_write_obj(surf.get(), obj_path, normalize_uv, align_grid, keep_percent, clean_surface, clean_sigma_k, inpaint_holes);

    return EXIT_SUCCESS;
}
