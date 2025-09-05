// vc_tifxyz2obj.cpp
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/types/ChunkedTensor.hpp"

#include "z5/factory.hxx"
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <limits>


using json = nlohmann::json;

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

// Decimates a grid of points by keeping every nth point in both dimensions
// This reduces the total point count by approximately (1 - 1/stride²)
// For stride=3, this keeps ~11% of points (close to the 10% target)
static cv::Mat_<cv::Vec3f> decimate_grid(const cv::Mat_<cv::Vec3f>& points, int iterations = 1)
{
    if (iterations <= 0) return points.clone();
    
    cv::Mat_<cv::Vec3f> result = points.clone();
    
    for (int iter = 0; iter < iterations; ++iter) {
        // Use stride of 3 to achieve ~89% reduction (keeping ~11%)
        const int stride = 3;
        
        // Calculate new dimensions
        int new_rows = (result.rows + stride - 1) / stride;
        int new_cols = (result.cols + stride - 1) / stride;
        
        cv::Mat_<cv::Vec3f> decimated(new_rows, new_cols);
        
        // Sample every stride-th point
        for (int j = 0; j < new_rows; ++j) {
            for (int i = 0; i < new_cols; ++i) {
                int src_j = j * stride;
                int src_i = i * stride;
                
                // Ensure we don't go out of bounds
                if (src_j < result.rows && src_i < result.cols) {
                    decimated(j, i) = result(src_j, src_i);
                } else {
                    // Handle edge case - use the last valid point
                    src_j = std::min(src_j, result.rows - 1);
                    src_i = std::min(src_i, result.cols - 1);
                    decimated(j, i) = result(src_j, src_i);
                }
            }
        }
        
        result = decimated;
        
        std::cout << "Decimation iteration " << (iter + 1) << ": " 
                  << "reduced to " << new_rows << " x " << new_cols 
                  << " (" << (new_rows * new_cols) << " points)" << std::endl;
    }
    
    return result;
}

// ---------------------------------------------------------------------------

// Cleans the surface by removing points that are spatial outliers
static cv::Mat_<cv::Vec3f> clean_surface_outliers(const cv::Mat_<cv::Vec3f>& points, float distance_threshold = 5.0f)
{
    cv::Mat_<cv::Vec3f> cleaned = points.clone();
    
    // Calculate average neighbor distance to determine outlier threshold
    std::vector<float> all_neighbor_dists;
    
    // First pass: collect all neighbor distances to compute statistics
    for (int j = 0; j < points.rows; ++j) {
        for (int i = 0; i < points.cols; ++i) {
            if (points(j, i)[0] == -1) continue;
            
            cv::Vec3f center = points(j, i);
            
            // Check 8-connected neighbors
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    
                    int ny = j + dy;
                    int nx = i + dx;
                    
                    if (ny >= 0 && ny < points.rows && nx >= 0 && nx < points.cols) {
                        if (points(ny, nx)[0] != -1) {
                            cv::Vec3f neighbor = points(ny, nx);
                            float dist = cv::norm(center - neighbor);
                            if (std::isfinite(dist) && dist > 0) {
                                all_neighbor_dists.push_back(dist);
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Calculate median and MAD for robust statistics
    float median_dist = 0;
    float mad = 0;
    
    if (!all_neighbor_dists.empty()) {
        std::sort(all_neighbor_dists.begin(), all_neighbor_dists.end());
        median_dist = all_neighbor_dists[all_neighbor_dists.size() / 2];
        
        // Calculate MAD
        std::vector<float> abs_deviations;
        for (float d : all_neighbor_dists) {
            abs_deviations.push_back(std::abs(d - median_dist));
        }
        std::sort(abs_deviations.begin(), abs_deviations.end());
        mad = abs_deviations[abs_deviations.size() / 2];
    }
    
    // Threshold is median + distance_threshold * scaled MAD
    float threshold = median_dist + distance_threshold * (mad / 0.6745f);
    
    std::cout << "Outlier detection statistics:" << std::endl;
    std::cout << "  Median neighbor distance: " << median_dist << std::endl;
    std::cout << "  MAD: " << mad << std::endl;
    std::cout << "  K (sigma multiplier): " << distance_threshold << std::endl;
    std::cout << "  Distance threshold: " << threshold << std::endl;
    
    // Second pass: identify and remove outliers
    for (int j = 0; j < points.rows; ++j) {
        for (int i = 0; i < points.cols; ++i) {
            if (points(j, i)[0] == -1) continue;
            
            cv::Vec3f center = points(j, i);
            float min_neighbor_dist = std::numeric_limits<float>::max();
            int neighbor_count = 0;
            
            // Check 8-connected neighbors
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    
                    int ny = j + dy;
                    int nx = i + dx;
                    
                    if (ny >= 0 && ny < points.rows && nx >= 0 && nx < points.cols) {
                        if (points(ny, nx)[0] != -1) {
                            cv::Vec3f neighbor = points(ny, nx);
                            float dist = cv::norm(center - neighbor);
                            if (std::isfinite(dist)) {
                                min_neighbor_dist = std::min(min_neighbor_dist, dist);
                                neighbor_count++;
                            }
                        }
                    }
                }
            }
            
            // Mark as outlier if:
            // 1. Has no neighbors at all, OR
            // 2. Closest neighbor is beyond threshold
            if (neighbor_count == 0 || (min_neighbor_dist > threshold && threshold > 0)) {
                cleaned(j, i) = cv::Vec3f(-1, -1, -1);
            }
        }
    }
    
    // Count removed points
    int removed_count = 0;
    for (int j = 0; j < points.rows; ++j) {
        for (int i = 0; i < points.cols; ++i) {
            if (points(j, i)[0] != -1 && cleaned(j, i)[0] == -1) {
                removed_count++;
            }
        }
    }
    
    std::cout << "Surface cleaning: removed " << removed_count << " outlier points" << std::endl;
    
    return cleaned;
}

// ---------------------------------------------------------------------------

// Adds vertex/texcoord/normal for grid location (y,x) if not already added.
// Returns the (1-based) OBJ index for this vertex (and matching vt/vn).
static int get_add_vertex(std::ofstream& out,
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

static void surf_write_obj(QuadSurface *surf, const std::filesystem::path &out_fn, bool normalize_uv, bool align_grid, int decimate_iterations, bool clean_surface, float clean_sigma_k)
{
    cv::Mat_<cv::Vec3f> points = surf->rawPoints();
    
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
    
    // Apply decimation if requested
    if (decimate_iterations > 0) {
        std::cout << "Original grid: " << points.rows << " x " << points.cols 
                  << " (" << (points.rows * points.cols) << " points)" << std::endl;
        points = decimate_grid(points, decimate_iterations);
    }
    
    cv::Mat_<int> idxs(points.size(), -1);

    std::ofstream out(out_fn);
    if (!out) {
        std::cerr << "Failed to open for write: " << out_fn << "\n";
        return;
    }
    out << std::fixed << std::setprecision(6);

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

    int v_idx = 1;
    for (int j = 0; j < points.rows - 1; ++j)
        for (int i = 0; i < points.cols - 1; ++i)
            if (loc_valid(points, cv::Vec2d(j, i)))
            {
                const int c00 = get_add_vertex(out, points, normals, idxs, v_idx, {j,   i  }, normalize_uv, uv_fac_x, uv_fac_y);
                const int c01 = get_add_vertex(out, points, normals, idxs, v_idx, {j,   i+1}, normalize_uv, uv_fac_x, uv_fac_y);
                const int c10 = get_add_vertex(out, points, normals, idxs, v_idx, {j+1, i  }, normalize_uv, uv_fac_x, uv_fac_y);
                const int c11 = get_add_vertex(out, points, normals, idxs, v_idx, {j+1, i+1}, normalize_uv, uv_fac_x, uv_fac_y);
                // faces unchanged: use same index for v/vt/vn
                out << "f " << c10 << "/" << c10 << "/" << c10 << " "
                           << c00 << "/" << c00 << "/" << c00 << " "
                           << c01 << "/" << c01 << "/" << c01 << '\n';

                out << "f " << c10 << "/" << c10 << "/" << c10 << " "
                           << c01 << "/" << c01 << "/" << c01 << " "
                           << c11 << "/" << c11 << "/" << c11 << '\n';
            }
}

int main(int argc, char *argv[])
{
    if (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        std::cout << "usage: " << argv[0] << " <tiffxyz> <obj> [--normalize-uv] [--align-grid] [--decimate [iterations]] [--clean [K]]\n"
                  << "  --normalize-uv : Normalize UVs to [0,1] range\n"
                  << "  --align-grid   : Align grid Z only (flatten Z per row)\n"
                  << "  --decimate [n] : Reduce points by ~90% per iteration (default n=1)\n"
                  << "  --clean [K]    : Remove outlier points far from surface using robust distance threshold; K is sigma multiplier (default 5.0)\n";
        return EXIT_SUCCESS;
    }

    if (argc < 3) {
    std::cerr << "error: too few arguments\n"
                  << "usage: " << argv[0] << " <tiffxyz> <obj> [--normalize-uv] [--align-grid] [--decimate [iterations]] [--clean [K]]\n";
        return EXIT_FAILURE;
    }

    bool normalize_uv = false;
    bool align_grid = false;
    bool clean_surface = false;
    float clean_sigma_k = 5.0f; // default K
    int decimate_iterations = 0;
    
    // Parse optional arguments
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--normalize-uv") {
            normalize_uv = true;
        } else if (arg == "--align-grid") {
            align_grid = true;
        } else if (arg == "--decimate") {
            decimate_iterations = 1; // Default to 1 iteration
            
            // Check if next argument is a number (iterations)
            if (i + 1 < argc) {
                std::string next = argv[i + 1];
                try {
                    int iters = std::stoi(next);
                    if (iters > 0) {
                        decimate_iterations = iters;
                        ++i; // Skip the number argument
                    }
                } catch (...) {
                    // Not a number, continue with default of 1
                }
            }
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

    QuadSurface *surf = nullptr;
    try {
        surf = load_quad_from_tifxyz(seg_path);
    }
    catch (...) {
        std::cerr << "error when loading: " << seg_path << std::endl;
        return EXIT_FAILURE;
    }

    surf_write_obj(surf, obj_path, normalize_uv, align_grid, decimate_iterations, clean_surface, clean_sigma_k);

    delete surf;
    return EXIT_SUCCESS;
}
