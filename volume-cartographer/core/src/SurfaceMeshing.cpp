#include "vc/core/util/SurfaceMeshing.hpp"

#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace vc::meshing {

int decimationStride(float ratio)
{
    if (ratio <= 0.0f || ratio >= 1.0f) return 1;
    // For ratio r, we want 1/s² = r, so s = 1/sqrt(r)
    return std::max(2, static_cast<int>(std::round(1.0f / std::sqrt(ratio))));
}

cv::Mat_<cv::Vec3f> decimateGrid(const cv::Mat_<cv::Vec3f>& points, int stride)
{
    if (stride <= 1) return points.clone();
    cv::Mat_<cv::Vec3f> decimated((points.rows + stride - 1) / stride, (points.cols + stride - 1) / stride);
    for (int j = 0; j < decimated.rows; ++j)
        for (int i = 0; i < decimated.cols; ++i) decimated(j, i) = points(j * stride, i * stride);
    return decimated;
}

GridMesh meshGrid(const cv::Mat_<cv::Vec3f>& points)
{
    GridMesh mesh;
    cv::Mat_<int> idxs(points.size(), -1);
    const auto vertex = [&](int row, int col) {
        int& idx = idxs(row, col);
        if (idx == -1) {
            idx = static_cast<int>(mesh.positions.size());
            mesh.positions.push_back(points(row, col));
            mesh.cells.emplace_back(row, col);
        }
        return idx;
    };
    for (int j = 0; j < points.rows - 1; ++j)
        for (int i = 0; i < points.cols - 1; ++i)
            if (loc_valid(points, cv::Vec2d(j, i))) {
                const int c00 = vertex(j, i);
                const int c01 = vertex(j, i + 1);
                const int c10 = vertex(j + 1, i);
                const int c11 = vertex(j + 1, i + 1);
                mesh.faces.emplace_back(c10, c00, c01);
                mesh.faces.emplace_back(c10, c01, c11);
            }
    return mesh;
}

LiftedUVs liftGridUVs(const GridMesh& fine, const cv::Mat_<cv::Vec3f>& coarsePoints, const GridMesh& coarse,
                      const std::vector<cv::Vec2d>& coarseUVs, int stride)
{
    if (coarseUVs.size() != coarse.cells.size()) throw std::invalid_argument("one UV per coarse vertex");
    cv::Mat_<int> index(coarsePoints.size(), -1);
    for (std::size_t i = 0; i < coarse.cells.size(); ++i) index(coarse.cells[i]) = static_cast<int>(i);
    const auto uvOf = [&](int r, int c) { return coarseUVs[index(r, c)]; };

    LiftedUVs lifted;
    lifted.uvs.resize(fine.cells.size());
    lifted.valid.assign(fine.cells.size(), 0);
    const int rows = coarsePoints.rows, cols = coarsePoints.cols;
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(fine.cells.size()); ++i) {
        const double gr = double(fine.cells[i][0]) / stride;
        const double gc = double(fine.cells[i][1]) / stride;
        const int qr = std::clamp(int(std::floor(gr)), 0, std::max(0, rows - 2));
        const int qc = std::clamp(int(std::floor(gc)), 0, std::max(0, cols - 2));
        if (rows > 1 && cols > 1 && loc_valid(coarsePoints, cv::Vec2d(qr, qc))) {
            // The quad's faces are (r+1,c)(r,c)(r,c+1) below the diagonal u+v=1
            // and (r+1,c)(r,c+1)(r+1,c+1) above it.
            const double u = std::clamp(gc - qc, 0.0, 1.0);
            const double v = std::clamp(gr - qr, 0.0, 1.0);
            const cv::Vec2d uv = u + v <= 1.0
                ? (1.0 - u - v) * uvOf(qr, qc) + u * uvOf(qr, qc + 1) + v * uvOf(qr + 1, qc)
                : (u + v - 1.0) * uvOf(qr + 1, qc + 1) + (1.0 - v) * uvOf(qr, qc + 1) + (1.0 - u) * uvOf(qr + 1, qc);
            lifted.uvs[i] = cv::Vec2f(float(uv[0]), float(uv[1]));
            lifted.valid[i] = 1;
            continue;
        }
        double best = std::numeric_limits<double>::infinity();
        const int cr = int(std::lround(gr)), cc = int(std::lround(gc));
        for (int r = std::max(0, cr - 2); r <= std::min(rows - 1, cr + 2); ++r)
            for (int c = std::max(0, cc - 2); c <= std::min(cols - 1, cc + 2); ++c) {
                if (index(r, c) < 0) continue;
                const double d = (r - gr) * (r - gr) + (c - gc) * (c - gc);
                if (d < best) {
                    best = d;
                    const cv::Vec2d uv = uvOf(r, c);
                    lifted.uvs[i] = cv::Vec2f(float(uv[0]), float(uv[1]));
                    lifted.valid[i] = 1;
                }
            }
    }
    return lifted;
}

bool readUvObj(const std::string& path, UvMesh& mesh)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Cannot open OBJ file: " << path << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#')
            continue;

        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            cv::Vec3f v;
            iss >> v[0] >> v[1] >> v[2];
            mesh.vertices.push_back(v);
        }
        else if (prefix == "vt") {
            cv::Vec2f uv;
            iss >> uv[0] >> uv[1];
            mesh.uvs.push_back(uv);
        }
        else if (prefix == "f") {
            UvFace face;
            std::string vertex_str;
            int idx = 0;

            while (iss >> vertex_str && idx < 3) {
                // Parse vertex/texture/normal indices
                std::replace(vertex_str.begin(), vertex_str.end(), '/', ' ');
                std::istringstream viss(vertex_str);

                viss >> face.v[idx];
                face.v[idx]--; // Convert to 0-based

                if (viss >> face.vt[idx]) {
                    face.vt[idx]--; // Convert to 0-based
                } else {
                    face.vt[idx] = -1;
                }

                idx++;
            }

            if (idx == 3) {
                mesh.faces.push_back(face);
            }
        }
    }

    file.close();

    std::cout << "Loaded OBJ file:" << std::endl;
    std::cout << "  Vertices: " << mesh.vertices.size() << std::endl;
    std::cout << "  UVs: " << mesh.uvs.size() << std::endl;
    std::cout << "  Faces: " << mesh.faces.size() << std::endl;

    return !mesh.vertices.empty() && !mesh.faces.empty() && !mesh.uvs.empty();
}

namespace {
// Sub-pixel resolution for fixed-point vertex snapping (8 fractional bits).
constexpr long long SUBPIXEL = 256;

// Twice the signed area of triangle (A, B, P): (B-A) x (P-A). Exact in int64.
long long edgeFn(long long ax, long long ay,
                 long long bx, long long by,
                 long long px, long long py) {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

// Top-left edge predicate for the fill rule. Antisymmetric under edge
// reversal, so the two triangles sharing an edge make opposite choices and
// each on-edge pixel is claimed exactly once.
bool isTopLeft(long long dx, long long dy) {
    return dy < 0 || (dy == 0 && dx < 0);
}
}

void UvGridRasterizer::setMesh(std::vector<cv::Vec3f> vertices_, std::vector<cv::Vec2f> uvs_, std::vector<UvFace> faces_)
{
    vertices = std::move(vertices_);
    uvs = std::move(uvs_);
    faces = std::move(faces_);
}

bool UvGridRasterizer::setGridUV(std::vector<cv::Vec2f> cells)
{
    grid_uv.clear();
    if (cells.size() != vertices.size()) return false;
    grid_uv = std::move(cells);
    return true;
}

void UvGridRasterizer::determineGridDimensions(float stretch_factor) {
    // Find UV bounds from all UVs used in faces
    uv_min = cv::Vec2f(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    uv_max = cv::Vec2f(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());

    for (const auto& face : faces) {
        for (int i = 0; i < 3; i++) {
            if (face.vt[i] >= 0 && face.vt[i] < (int)uvs.size()) {
                cv::Vec2f uv = uvs[face.vt[i]];
                uv_min[0] = std::min(uv_min[0], uv[0]);
                uv_min[1] = std::min(uv_min[1], uv[1]);
                uv_max[0] = std::max(uv_max[0], uv[0]);
                uv_max[1] = std::max(uv_max[1], uv[1]);
            }
        }
    }

    std::cout << "UV bounds: [" << uv_min[0] << ", " << uv_min[1] << "] to [" 
              << uv_max[0] << ", " << uv_max[1] << "]" << std::endl;

    // Build raw grid (pre-decimation) from UV range and stretch factor
    const cv::Vec2f uv_range = uv_max - uv_min;
    const int gw_raw = std::max(2, static_cast<int>(std::ceil(uv_range[0] * stretch_factor)) + 1);
    const int gh_raw = std::max(2, static_cast<int>(std::ceil(uv_range[1] * stretch_factor)) + 1);

    // Apply uniform decimation / capping to the resolution (NOT to the scale)
    const auto apply_decimation = [](int n, double d) {
        if (d <= 1.0) return n;
        // keep endpoints: 1 + floor((n-1)/d)
        const double eff = std::max(1.0, d);
        return std::max(2, static_cast<int>(std::floor((n - 1) / eff)) + 1);
    };

    double decim = std::max(1.0, static_cast<double>(uv_downsample));
    if (grid_cap_pixels > 0) {
        const long double raw_px = static_cast<long double>(gw_raw) * static_cast<long double>(gh_raw);
        if (raw_px > static_cast<long double>(grid_cap_pixels)) {
            const double need = std::sqrt(static_cast<double>(raw_px / static_cast<long double>(grid_cap_pixels)));
            decim = std::max(decim, std::ceil(need));
        }
    }
    const int gw_dec = apply_decimation(gw_raw, decim);
    const int gh_dec = apply_decimation(gh_raw, decim);
    grid_size[0] = gw_dec;
    grid_size[1] = gh_dec;

    // Effective decimation per axis (accounts for rounding at endpoints)
    const double eff_decim_x = (gw_dec > 1) ? (double)(gw_raw - 1) / (double)(gw_dec - 1) : 1.0;
    const double eff_decim_y = (gh_dec > 1) ? (double)(gh_raw - 1) / (double)(gh_dec - 1) : 1.0;

    // When the source tifxyz scale is known, size the output grid to the
    // source sampling density and adopt its scale verbatim. vc_tifxyz2obj
    // wrote UVs as grid_idx*(1/src_scale), so the source cell count per axis
    // is ~uv_range*src_scale + 1. This keeps the flattened tifxyz at the
    // input's resolution/scale instead of metric mode's 1 cell / UV unit
    // (which inflated the grid by ~1/src_scale per axis). Decimation/caps
    // still apply on top, never finer than source.
    if (src_scale[0] > 0.f && src_scale[1] > 0.f) {
        int gw_src = std::max(2, static_cast<int>(std::lround(uv_range[0] * src_scale[0])) + 1);
        int gh_src = std::max(2, static_cast<int>(std::lround(uv_range[1] * src_scale[1])) + 1);
        // Honor any further user decimation / pixel cap on top of source density.
        gw_src = apply_decimation(gw_src, decim);
        gh_src = apply_decimation(gh_src, decim);
        grid_size[0] = gw_src;
        grid_size[1] = gh_src;
        scale[0] = src_scale[0];
        scale[1] = src_scale[1];
        std::cout << "Source-scale mode: grid " << grid_size[0] << " x " << grid_size[1]
                  << "  scale: " << scale[0] << ", " << scale[1]
                  << "  (matched to input tifxyz)" << std::endl;
    } else if (uv_is_metric) {
        // --- Preserve physical pixel size (scale) from the RAW grid ---
        // du_raw/dv_raw are UV units per pixel *before* decimation.
        const float du_raw = (gw_raw > 1) ? uv_range[0] / float(gw_raw - 1) : 0.f;
        const float dv_raw = (gh_raw > 1) ? uv_range[1] / float(gh_raw - 1) : 0.f;
        scale[0] = du_raw * uv_to_obj; // OBJ units per pixel (unchanged by decimation)
        scale[1] = dv_raw * uv_to_obj;

        std::cout << "UV-metric mode: grid " << grid_size[0] << " x " << grid_size[1]
                  << "  scale(OBJ units): " << scale[0] << ", " << scale[1] << std::endl;
        if (decim > 1.0) {
            std::cout << "  (applied UV decimation ~" << decim << "x per axis; "
                      << "effective decim [" << eff_decim_x << ", " << eff_decim_y << "]; "
                      << "preserved per-pixel physical scale)\n";
        }
    } else {
        // --- Legacy non-metric: measure from 3D on decimated grid, then compensate ---
        std::cout << "Creating preliminary grid " << grid_size[0] << " x " << grid_size[1]
                  << " to measure scale from 3D..." << std::endl;

        cv::Mat_<cv::Vec3f> preliminary_points(grid_size[1], grid_size[0], cv::Vec3f(-1, -1, -1));
        for (const auto& face : faces) {
            rasterizeTriangle(preliminary_points, face);
        }
        // This fills 'scale' with the distance between adjacent samples on the DECIMATED grid
        calculateScaleFromGrid(preliminary_points);

        // Compensate by the effective decimation so that scale matches the raw-grid pixel size
        if (scale[0] > 0) scale[0] = static_cast<float>(scale[0] / eff_decim_x);
        if (scale[1] > 0) scale[1] = static_cast<float>(scale[1] / eff_decim_y);

        // Keep original behavior for the final log (do not fight it further)
        const cv::Vec2f measured = scale;
        grid_size[0] = std::max(2, static_cast<int>(std::round(measured[0] * stretch_factor)) + 1);
        grid_size[1] = std::max(2, static_cast<int>(std::round(measured[1] * stretch_factor)) + 1);

        std::cout << "Final grid: " << grid_size[0] << " x " << grid_size[1] << std::endl;
        std::cout << "Scale(OBJ units; compensated to raw-pixel spacing): "
                  << measured[0] << ", " << measured[1] << std::endl;
    }

    // rasterizeTriangle evaluates edge functions in int64 on fixed-point
    // coordinates up to (grid-1)*SUBPIXEL. Keep 2*L^2 below INT64_MAX so the
    // signed-area computation cannot overflow. The limit (~8.4M px/axis) is
    // far above any real tifxyz grid; tripping it means the inputs are wrong.
    const long long max_dim = std::max(grid_size[0], grid_size[1]);
    if (max_dim * SUBPIXEL >= 2'000'000'000LL) {
        throw std::runtime_error(
            "grid dimension too large for exact int64 rasterization: "
            + std::to_string(max_dim));
    }
}


QuadSurface* UvGridRasterizer::createQuadSurface(float mesh_units) {
    // Create points matrix initialized with invalid values
    cv::Mat_<cv::Vec3f>* points = new cv::Mat_<cv::Vec3f>(grid_size[1], grid_size[0], cv::Vec3f(-1, -1, -1));

    // Approval accumulator (unapproved = black) when resampling is active.
    if (hasApproval()) {
        out_approval = cv::Mat_<cv::Vec3b>(grid_size[1], grid_size[0], cv::Vec3b(0, 0, 0));
    }

    // Rasterize triangles onto the grid
    for (const auto& face : faces) {
        rasterizeTriangle(*points, face);
    }

    // Count valid points
    int valid_count = 0;
    for (int y = 0; y < grid_size[1]; y++) {
        for (int x = 0; x < grid_size[0]; x++) {
            if (vc::geometry::validSurfacePoint((*points)(y, x))) {
                valid_count++;
            }
        }
    }

    std::cout << "Valid grid points: " << valid_count << " / " << (grid_size[0] * grid_size[1]) 
              << " (" << (100.0f * valid_count / (grid_size[0] * grid_size[1])) << "%)" << std::endl;

    // Scale is currently in OBJ units. Convert to micrometers now.
    if (valid_count == 0) {
        std::cerr << "Warning: no valid grid points were rasterized." << std::endl;
    }
    const bool src_scale_mode = (src_scale[0] > 0.f && src_scale[1] > 0.f);
    if (src_scale_mode) {
        // Scale was adopted verbatim from the source tifxyz; do not rescale.
        std::cout << "Scale (from source tifxyz): " << scale[0] << ", " << scale[1] << std::endl;
    } else if (uv_is_metric) {
        scale[0] *= mesh_units;
        scale[1] *= mesh_units;
        std::cout << "Scale from UV (micrometers): " << scale[0] << ", " << scale[1] << std::endl;
    } else {
        // Measure from 3D to preserve anisotropy and noise-robustness (already compensated in determineGridDimensions)
        calculateScaleFromGrid(*points, mesh_units);
    }

    return new QuadSurface(points, scale);
}


void UvGridRasterizer::calculateScaleFromGrid(const cv::Mat_<cv::Vec3f>& points, float mesh_units) {
    // Based on vc_segmentation_scales from Slicing.cpp
    double sum_x = 0;
    double sum_y = 0;
    int count = 0;

    // Skip borders (10% on each side) to avoid artifacts
    int jmin = static_cast<int>(points.rows * 0.1) + 1;
    int jmax = static_cast<int>(points.rows * 0.9);
    int imin = static_cast<int>(points.cols * 0.1) + 1;
    int imax = static_cast<int>(points.cols * 0.9);
    int step = 4;

    // For small grids, use all points
    if (points.rows < 20 || points.cols < 20) {
        jmin = 1;
        jmax = points.rows;
        imin = 1;
        imax = points.cols;
        step = 1;
    }

    // Calculate average distance between adjacent points
    for (int j = jmin; j < jmax; j += step) {
        for (int i = imin; i < imax; i += step) {
            // Skip invalid points
            if (!vc::geometry::validSurfacePoint(points(j, i)) || !vc::geometry::validSurfacePoint(points(j, i-1)) || !vc::geometry::validSurfacePoint(points(j-1, i)))
                continue;

            // Distance to neighbor in X direction
            cv::Vec3f v = points(j, i) - points(j, i-1);
            const double dist_x_sq = v.dot(v);
            if (dist_x_sq > 0) {
                sum_x += std::sqrt(dist_x_sq);
            }

            // Distance to neighbor in Y direction
            v = points(j, i) - points(j-1, i);
            const double dist_y_sq = v.dot(v);
            if (dist_y_sq > 0) {
                sum_y += std::sqrt(dist_y_sq);
            }
            count++;
        }
    }

    if (count > 0 && sum_x > 0 && sum_y > 0) {
        // Scale is the average distance between points, adjusted by mesh units
        scale[0] = static_cast<float>((sum_x / count) * mesh_units);
        scale[1] = static_cast<float>((sum_y / count) * mesh_units);
    } else {
        // Fallback to UV-based scale if we couldn't calculate from grid
        std::cerr << "Warning: Could not calculate scale from grid, using UV-based fallback" << std::endl;
        // scale already set in determineGridDimensions
    }

    std::cout << "Calculated scale factors from grid: " << scale[0] << ", " << scale[1] << " micrometers" << std::endl;
}

void UvGridRasterizer::rasterizeTriangle(cv::Mat_<cv::Vec3f>& points, const UvFace& face) {
    // Validate indices
    for (int k = 0; k < 3; ++k) {
        if (face.v[k] < 0 || face.v[k] >= (int)vertices.size()) {
            return; // skip invalid vertex indices
        }
        if (face.vt[k] < 0 || face.vt[k] >= (int)uvs.size()) {
            return; // skip faces with missing/invalid UVs
        }
    }
    // Triangle 3D positions (reordered together with the UVs below to
    // enforce a consistent winding for the fill rule).
    cv::Vec3f pos[3] = { vertices[face.v[0]],
                         vertices[face.v[1]],
                         vertices[face.v[2]] };
    int vidx[3] = { face.v[0], face.v[1], face.v[2] }; // for approval grid_uv lookup

    // Map UVs into grid space in double, then snap each vertex to a
    // fixed-point sub-pixel grid. Integer edge functions then make the
    // inclusion test exact (no float rounding, identical on every
    // architecture); the top-left fill rule assigns each pixel that lands
    // on a shared edge to exactly one triangle, so a grid-aligned surface
    // is covered without the periodic gaps a float barycentric test left.
    const double rx = std::max<double>(uv_max[0] - uv_min[0], 1e-12);
    const double ry = std::max<double>(uv_max[1] - uv_min[1], 1e-12);
    long long X[3], Y[3];
    double gx[3], gy[3];
    for (int k = 0; k < 3; ++k) {
        const cv::Vec2f uv = uvs[face.vt[k]];
        gx[k] = (double(uv[0]) - uv_min[0]) / rx * (grid_size[0] - 1);
        gy[k] = (double(uv[1]) - uv_min[1]) / ry * (grid_size[1] - 1);
        X[k] = std::llround(gx[k] * SUBPIXEL);
        Y[k] = std::llround(gy[k] * SUBPIXEL);
    }

    // Twice the signed area. Zero -> degenerate (covers nothing).
    // Normalize to CCW so the top-left rule has a consistent orientation.
    long long area2 = edgeFn(X[0], Y[0], X[1], Y[1], X[2], Y[2]);
    if (area2 == 0) {
        return;
    }
    if (area2 < 0) {
        std::swap(X[1], X[2]); std::swap(Y[1], Y[2]);
        std::swap(pos[1], pos[2]); std::swap(vidx[1], vidx[2]);
        area2 = -area2;
    }

    // Top-left bias per edge: a pixel exactly on an edge (w == 0) is
    // included only when that edge is top-left. Folding the bias into the
    // comparison turns the rule into a single (w + bias) >= 0 test.
    const long long bias0 = isTopLeft(X[2] - X[1], Y[2] - Y[1]) ? 0 : -1; // edge B->C
    const long long bias1 = isTopLeft(X[0] - X[2], Y[0] - Y[2]) ? 0 : -1; // edge C->A
    const long long bias2 = isTopLeft(X[1] - X[0], Y[1] - Y[0]) ? 0 : -1; // edge A->B

    // Bounding box on the pixel grid.
    const int min_x = std::max(0, static_cast<int>(std::floor(std::min({gx[0], gx[1], gx[2]}))));
    const int max_x = std::min(grid_size[0] - 1, static_cast<int>(std::ceil(std::max({gx[0], gx[1], gx[2]}))));
    const int min_y = std::max(0, static_cast<int>(std::floor(std::min({gy[0], gy[1], gy[2]}))));
    const int max_y = std::min(grid_size[1] - 1, static_cast<int>(std::ceil(std::max({gy[0], gy[1], gy[2]}))));

    const double inv_area = 1.0 / static_cast<double>(area2);
    for (int y = min_y; y <= max_y; y++) {
        const long long Py = static_cast<long long>(y) * SUBPIXEL;
        for (int x = min_x; x <= max_x; x++) {
            const long long Px = static_cast<long long>(x) * SUBPIXEL;
            const long long w0 = edgeFn(X[1], Y[1], X[2], Y[2], Px, Py); // weight of vertex 0
            const long long w1 = edgeFn(X[2], Y[2], X[0], Y[0], Px, Py); // weight of vertex 1
            const long long w2 = edgeFn(X[0], Y[0], X[1], Y[1], Px, Py); // weight of vertex 2
            if ((w0 + bias0) < 0 || (w1 + bias1) < 0 || (w2 + bias2) < 0) {
                continue;
            }
            if (vc::geometry::validSurfacePoint(points(y, x))) {
                continue; // first triangle wins
            }
            const float b0 = static_cast<float>(w0 * inv_area);
            const float b1 = static_cast<float>(w1 * inv_area);
            const float b2 = static_cast<float>(w2 * inv_area);
            points(y, x) = b0 * pos[0] + b1 * pos[1] + b2 * pos[2];
            if (hasApproval()) {
                const cv::Vec2f cr = b0 * grid_uv[vidx[0]]
                                   + b1 * grid_uv[vidx[1]]
                                   + b2 * grid_uv[vidx[2]];
                out_approval(y, x) = sampleSrcApproval(cr[0], cr[1]);
            }
        }
    }
}


// Nearest-neighbor sample of the source approval mask at original grid
// (col,row). Returns BGR; 1-channel legacy masks map nonzero -> white.
cv::Vec3b UvGridRasterizer::sampleSrcApproval(float colf, float rowf) const {
    int col = static_cast<int>(std::lround(colf));
    int row = static_cast<int>(std::lround(rowf));
    col = std::clamp(col, 0, src_approval.cols - 1);
    row = std::clamp(row, 0, src_approval.rows - 1);
    if (src_approval.channels() == 1) {
        return src_approval.at<uint8_t>(row, col) > 0 ? cv::Vec3b(255, 255, 255)
                                                      : cv::Vec3b(0, 0, 0);
    }
    return src_approval.at<cv::Vec3b>(row, col);
}


void StepUvRasterizer::setMesh(std::vector<cv::Vec3f> v, std::vector<cv::Vec2f> t, std::vector<UvFace> f)
{
    vertices = std::move(v);
    uvs = std::move(t);
    faces = std::move(f);
}

void StepUvRasterizer::determineGridDimensions(float step, ScaleMethod method, cv::Vec2d manual_scale)
{
    step_size = step;

    // Find UV bounds from all UVs used in faces
    uv_min = cv::Vec2f(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    uv_max = cv::Vec2f(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());

    // Compute directional scale factors by analyzing UV-to-3D mapping gradients
    // For each triangle, compute the Jacobian of the mapping to get stretch in U and V
    double sum_scale_u = 0.0, sum_scale_v = 0.0;
    double sum_weight = 0.0;
    // per triangle norms and UV areas, for an area weighted median (robust to a few stretched triangles)
    std::vector<std::pair<double, double>> norms_u, norms_v;

    for (const auto& face : faces) {
        // Check valid indices
        bool valid = true;
        for (int i = 0; i < 3; i++) {
            if (face.v[i] < 0 || face.v[i] >= (int)vertices.size() ||
                face.vt[i] < 0 || face.vt[i] >= (int)uvs.size()) {
                valid = false;
                break;
            }
        }
        if (!valid) continue;

        // Update UV bounds
        for (int i = 0; i < 3; i++) {
            cv::Vec2f uv = uvs[face.vt[i]];
            uv_min[0] = std::min(uv_min[0], uv[0]);
            uv_min[1] = std::min(uv_min[1], uv[1]);
            uv_max[0] = std::max(uv_max[0], uv[0]);
            uv_max[1] = std::max(uv_max[1], uv[1]);
        }

        // Get triangle vertices and UVs
        cv::Vec3f p0 = vertices[face.v[0]];
        cv::Vec3f p1 = vertices[face.v[1]];
        cv::Vec3f p2 = vertices[face.v[2]];

        cv::Vec2f uv0 = uvs[face.vt[0]];
        cv::Vec2f uv1 = uvs[face.vt[1]];
        cv::Vec2f uv2 = uvs[face.vt[2]];

        // Compute edge vectors in UV and 3D
        cv::Vec2f e1_uv = uv1 - uv0;
        cv::Vec2f e2_uv = uv2 - uv0;
        cv::Vec3f e1_3d = p1 - p0;
        cv::Vec3f e2_3d = p2 - p0;

        // Compute UV triangle area for weighting
        double uv_cross = e1_uv[0] * e2_uv[1] - e1_uv[1] * e2_uv[0];
        double uv_area = 0.5 * std::abs(uv_cross);

        if (uv_area < 1e-12) continue;  // Skip degenerate triangles

        // Solve for the Jacobian: [dp/du, dp/dv] = [e1_3d, e2_3d] * inv([e1_uv, e2_uv]^T)
        // inv([a b; c d]) = 1/(ad-bc) * [d -b; -c a]
        double det = uv_cross;
        double inv_det = 1.0 / det;

        // dp/du = (e2_uv[1] * e1_3d - e1_uv[1] * e2_3d) / det
        // dp/dv = (-e2_uv[0] * e1_3d + e1_uv[0] * e2_3d) / det
        cv::Vec3f dp_du = (e2_uv[1] * e1_3d - e1_uv[1] * e2_3d) * inv_det;
        cv::Vec3f dp_dv = (-e2_uv[0] * e1_3d + e1_uv[0] * e2_3d) * inv_det;

        // Scale factors are the magnitudes of the gradients
        double scale_u = std::sqrt(dp_du.dot(dp_du));
        double scale_v = std::sqrt(dp_dv.dot(dp_dv));

        // Weight by triangle area
        sum_scale_u += scale_u * uv_area;
        sum_scale_v += scale_v * uv_area;
        sum_weight += uv_area;
        norms_u.emplace_back(scale_u, uv_area);
        norms_v.emplace_back(scale_v, uv_area);
    }

    // Scale factors: area weighted median (default), area weighted mean (the behaviour before
    // the median), or the value given on the command line
    auto weighted_median = [](std::vector<std::pair<double, double>>& v, double fallback) {
        if (v.empty()) return fallback;
        std::sort(v.begin(), v.end());
        double total = 0.0;
        for (const auto& e : v) total += e.second;
        double acc = 0.0;
        for (const auto& e : v) {
            acc += e.second;
            if (acc >= 0.5 * total) return e.first;
        }
        return v.back().first;
    };
    const double mean_scale_u = (sum_weight > 0) ? sum_scale_u / sum_weight : 1.0;
    const double mean_scale_v = (sum_weight > 0) ? sum_scale_v / sum_weight : 1.0;
    double avg_scale_u = mean_scale_u, avg_scale_v = mean_scale_v;
    if (method == ScaleMethod::Median) {
        avg_scale_u = weighted_median(norms_u, mean_scale_u);
        avg_scale_v = weighted_median(norms_v, mean_scale_v);
    } else if (method == ScaleMethod::Manual) {
        avg_scale_u = manual_scale[0];
        avg_scale_v = manual_scale[1];
    }

    cv::Vec2f uv_range = uv_max - uv_min;

    std::cout << "UV bounds: [" << uv_min[0] << ", " << uv_min[1] << "] to ["
              << uv_max[0] << ", " << uv_max[1] << "]" << std::endl;
    std::cout << "Directional scales ("
              << (method == ScaleMethod::Median ? "median" : method == ScaleMethod::Mean ? "mean" : "manual")
              << "): U=" << avg_scale_u << ", V=" << avg_scale_v << std::endl;

    // Compute full voxel resolution
    double full_res_u = uv_range[0] * avg_scale_u;
    double full_res_v = uv_range[1] * avg_scale_v;

    // Store for UV-to-grid mapping in rasterization
    full_resolution = cv::Vec2d(full_res_u, full_res_v);

    // Downsample by step_size for actual grid/tif dimensions
    grid_size[0] = static_cast<int>(std::ceil(full_res_u / step_size)) + 1;
    grid_size[1] = static_cast<int>(std::ceil(full_res_v / step_size)) + 1;

    std::cout << "Full resolution: " << full_res_u << " x " << full_res_v << std::endl;
    std::cout << "Grid dimensions: " << grid_size[0] << " x " << grid_size[1] << std::endl;
    std::cout << "Step size: " << step_size << ", Scale: " << (1.0f / step_size) << std::endl;
}

QuadSurface* StepUvRasterizer::createQuadSurface()
{
    // Create points matrix initialized with invalid values
    cv::Mat_<cv::Vec3f>* points = new cv::Mat_<cv::Vec3f>(grid_size[1], grid_size[0], cv::Vec3f(-1, -1, -1));

    // Rasterize triangles onto the grid
    for (const auto& face : faces) {
        rasterizeTriangle(*points, face);
    }

    // Count valid points
    int valid_count = 0;
    for (int y = 0; y < grid_size[1]; y++) {
        for (int x = 0; x < grid_size[0]; x++) {
            if (vc::geometry::validSurfacePoint((*points)(y, x))) {
                valid_count++;
            }
        }
    }

    std::cout << "Valid grid points: " << valid_count << " / " << (grid_size[0] * grid_size[1])
              << " (" << (100.0f * valid_count / (grid_size[0] * grid_size[1])) << "%)" << std::endl;

    // Scale = 1/step_size (matching GrowPatch.cpp pattern)
    cv::Vec2f scale = {1.0f / step_size, 1.0f / step_size};

    return new QuadSurface(points, scale);
}

namespace {
cv::Vec3f stepBarycentric(const cv::Vec2f& p, const cv::Vec2f& a, const cv::Vec2f& b, const cv::Vec2f& c)
{
    cv::Vec2f v0 = c - a;
    cv::Vec2f v1 = b - a;
    cv::Vec2f v2 = p - a;

    float dot00 = v0.dot(v0);
    float dot01 = v0.dot(v1);
    float dot02 = v0.dot(v2);
    float dot11 = v1.dot(v1);
    float dot12 = v1.dot(v2);

    float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
    float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    return cv::Vec3f(1.0f - u - v, v, u);
}
}

void StepUvRasterizer::rasterizeTriangle(cv::Mat_<cv::Vec3f>& points, const UvFace& face) const
{
    // Get triangle vertices and UVs
    cv::Vec3f v0 = vertices[face.v[0]];
    cv::Vec3f v1 = vertices[face.v[1]];
    cv::Vec3f v2 = vertices[face.v[2]];

    cv::Vec2f uv0 = uvs[face.vt[0]];
    cv::Vec2f uv1 = uvs[face.vt[1]];
    cv::Vec2f uv2 = uvs[face.vt[2]];

    // Transform UVs to grid coordinates (downsampled)
    // 1. Map UV from [uv_min, uv_max] to full voxel resolution [0, full_resolution]
    // 2. Divide by step_size to get grid coordinates
    cv::Vec2f uv_range = uv_max - uv_min;

    // Map to full resolution, then downsample to grid coordinates
    uv0 = (uv0 - uv_min);
    uv0[0] = uv0[0] / uv_range[0] * full_resolution[0] / step_size;
    uv0[1] = uv0[1] / uv_range[1] * full_resolution[1] / step_size;

    uv1 = (uv1 - uv_min);
    uv1[0] = uv1[0] / uv_range[0] * full_resolution[0] / step_size;
    uv1[1] = uv1[1] / uv_range[1] * full_resolution[1] / step_size;

    uv2 = (uv2 - uv_min);
    uv2[0] = uv2[0] / uv_range[0] * full_resolution[0] / step_size;
    uv2[1] = uv2[1] / uv_range[1] * full_resolution[1] / step_size;

    // Find bounding box in grid coordinates
    int min_x = std::max(0, static_cast<int>(std::floor(std::min({uv0[0], uv1[0], uv2[0]}))) - 1);
    int max_x = std::min(grid_size[0] - 1, static_cast<int>(std::ceil(std::max({uv0[0], uv1[0], uv2[0]}))) + 1);
    int min_y = std::max(0, static_cast<int>(std::floor(std::min({uv0[1], uv1[1], uv2[1]}))) - 1);
    int max_y = std::min(grid_size[1] - 1, static_cast<int>(std::ceil(std::max({uv0[1], uv1[1], uv2[1]}))) + 1);

    // Rasterize triangle
    for (int y = min_y; y <= max_y; y++) {
        for (int x = min_x; x <= max_x; x++) {
            cv::Vec2f p(x, y);

            // Compute barycentric coordinates
            cv::Vec3f bary = stepBarycentric(p, uv0, uv1, uv2);

            // Check if point is inside triangle
            if (bary[0] >= 0 && bary[1] >= 0 && bary[2] >= 0) {
                // Interpolate 3D position
                cv::Vec3f pos = bary[0] * v0 + bary[1] * v1 + bary[2] * v2;

                // Only update if not already set (first triangle wins)
                if (!vc::geometry::validSurfacePoint(points(y, x))) {
                    points(y, x) = pos;
                }
            }
        }
    }
}


} // namespace vc::meshing
