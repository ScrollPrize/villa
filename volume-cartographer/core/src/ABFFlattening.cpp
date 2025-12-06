#include "vc/core/util/ABFFlattening.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <OpenABF/OpenABF.hpp>

#include <Eigen/Core>
#include <opencv2/imgproc.hpp>

#include <unordered_map>
#include <iostream>
#include <cmath>
#include <limits>

namespace vc {

// Type aliases for OpenABF
using HalfEdgeMesh = OpenABF::detail::ABF::Mesh<double>;
using ABF = OpenABF::ABFPlusPlus<double>;
using LSCM = OpenABF::AngleBasedLSCM<double, HalfEdgeMesh>;

/**
 * @brief Compute 3D surface area using triangulation of valid quads (parallelized)
 */
static double computeSurfaceArea3D(const QuadSurface& surface) {
    double area = 0.0;
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();

    #pragma omp parallel for collapse(2) reduction(+:area)
    for (int row = 0; row < points->rows - 1; ++row) {
        for (int col = 0; col < points->cols - 1; ++col) {
            const cv::Vec3f& p00 = (*points)(row, col);
            const cv::Vec3f& p01 = (*points)(row, col + 1);
            const cv::Vec3f& p10 = (*points)(row + 1, col);
            const cv::Vec3f& p11 = (*points)(row + 1, col + 1);

            // Skip invalid quads
            if (p00[0] == -1.f || p01[0] == -1.f || p10[0] == -1.f || p11[0] == -1.f)
                continue;

            // Triangle 1: p10, p00, p01
            cv::Vec3f e1 = p00 - p10;
            cv::Vec3f e2 = p01 - p10;
            cv::Vec3f cross1(
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0]
            );
            area += 0.5 * std::sqrt(cross1.dot(cross1));

            // Triangle 2: p10, p01, p11
            e1 = p01 - p10;
            e2 = p11 - p10;
            cv::Vec3f cross2(
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0]
            );
            area += 0.5 * std::sqrt(cross2.dot(cross2));
        }
    }
    return area;
}

/**
 * @brief Compute 2D area from UV coordinates (parallelized)
 */
static double computeArea2D(const cv::Mat_<cv::Vec2f>& uvs, const QuadSurface& surface) {
    double area = 0.0;
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();

    #pragma omp parallel for collapse(2) reduction(+:area)
    for (int row = 0; row < points->rows - 1; ++row) {
        for (int col = 0; col < points->cols - 1; ++col) {
            const cv::Vec3f& p00 = (*points)(row, col);
            const cv::Vec3f& p01 = (*points)(row, col + 1);
            const cv::Vec3f& p10 = (*points)(row + 1, col);
            const cv::Vec3f& p11 = (*points)(row + 1, col + 1);

            // Skip invalid quads
            if (p00[0] == -1.f || p01[0] == -1.f || p10[0] == -1.f || p11[0] == -1.f)
                continue;

            const cv::Vec2f& uv00 = uvs(row, col);
            const cv::Vec2f& uv01 = uvs(row, col + 1);
            const cv::Vec2f& uv10 = uvs(row + 1, col);
            const cv::Vec2f& uv11 = uvs(row + 1, col + 1);

            // Triangle 1: uv10, uv00, uv01
            cv::Vec2f e1 = uv00 - uv10;
            cv::Vec2f e2 = uv01 - uv10;
            double cross1 = e1[0] * e2[1] - e1[1] * e2[0];
            area += 0.5 * std::abs(cross1);

            // Triangle 2: uv10, uv01, uv11
            e1 = uv01 - uv10;
            e2 = uv11 - uv10;
            double cross2 = e1[0] * e2[1] - e1[1] * e2[0];
            area += 0.5 * std::abs(cross2);
        }
    }
    return area;
}

/**
 * @brief Downsample a grid by taking every Nth row and column
 *
 * @param grid Input grid
 * @param factor Downsample factor (2 = half, 4 = quarter, etc.)
 * @return Downsampled grid
 */
static cv::Mat_<cv::Vec3f> downsampleGrid(const cv::Mat_<cv::Vec3f>& grid, int factor) {
    if (factor <= 1) {
        return grid.clone();
    }

    int outRows = (grid.rows + factor - 1) / factor;
    int outCols = (grid.cols + factor - 1) / factor;

    cv::Mat_<cv::Vec3f> result(outRows, outCols);

    #pragma omp parallel for collapse(2)
    for (int outRow = 0; outRow < outRows; ++outRow) {
        for (int outCol = 0; outCol < outCols; ++outCol) {
            int inRow = outRow * factor;
            int inCol = outCol * factor;
            result(outRow, outCol) = grid(inRow, inCol);
        }
    }

    return result;
}

/**
 * @brief Upsample UV coordinates from coarse grid to fine grid using bilinear interpolation
 *
 * @param coarseUVs UV coordinates on downsampled grid
 * @param originalRows Original grid height
 * @param originalCols Original grid width
 * @param factor Downsample factor used
 * @return Upsampled UVs matching original grid dimensions
 */
static cv::Mat_<cv::Vec2f> upsampleUVs(const cv::Mat_<cv::Vec2f>& coarseUVs,
                                        int originalRows, int originalCols,
                                        int factor) {
    cv::Mat_<cv::Vec2f> result(originalRows, originalCols, cv::Vec2f(-1.f, -1.f));

    #pragma omp parallel for collapse(2)
    for (int row = 0; row < originalRows; ++row) {
        for (int col = 0; col < originalCols; ++col) {
            // Map to coarse grid coordinates (floating point)
            float coarseRowF = static_cast<float>(row) / factor;
            float coarseColF = static_cast<float>(col) / factor;

            // Get integer indices and fractional parts
            int r0 = static_cast<int>(coarseRowF);
            int c0 = static_cast<int>(coarseColF);
            int r1 = std::min(r0 + 1, coarseUVs.rows - 1);
            int c1 = std::min(c0 + 1, coarseUVs.cols - 1);

            float fr = coarseRowF - r0;
            float fc = coarseColF - c0;

            // Get the four corner UVs
            const cv::Vec2f& uv00 = coarseUVs(r0, c0);
            const cv::Vec2f& uv01 = coarseUVs(r0, c1);
            const cv::Vec2f& uv10 = coarseUVs(r1, c0);
            const cv::Vec2f& uv11 = coarseUVs(r1, c1);

            // Check if any corner is invalid
            if (uv00[0] == -1.f || uv01[0] == -1.f ||
                uv10[0] == -1.f || uv11[0] == -1.f) {
                // If any corner is invalid, try to use nearest valid sample
                if (uv00[0] != -1.f) {
                    result(row, col) = uv00;
                } else if (uv01[0] != -1.f) {
                    result(row, col) = uv01;
                } else if (uv10[0] != -1.f) {
                    result(row, col) = uv10;
                } else if (uv11[0] != -1.f) {
                    result(row, col) = uv11;
                }
                // Otherwise leave as invalid
                continue;
            }

            // Bilinear interpolation
            cv::Vec2f uv = (1 - fr) * (1 - fc) * uv00 +
                           (1 - fr) * fc * uv01 +
                           fr * (1 - fc) * uv10 +
                           fr * fc * uv11;
            result(row, col) = uv;
        }
    }

    return result;
}

/**
 * @brief Internal ABF++ flattening on the provided surface (no downsampling)
 */
static cv::Mat_<cv::Vec2f> abfFlattenInternal(const QuadSurface& surface, const ABFConfig& config) {
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    if (!points || points->empty()) {
        std::cerr << "ABF++: Empty surface" << std::endl;
        return cv::Mat_<cv::Vec2f>();
    }

    // Initialize UV output with invalid values
    cv::Mat_<cv::Vec2f> uvs(points->size(), cv::Vec2f(-1.f, -1.f));

    // Build half-edge mesh from valid quads
    auto hem = HalfEdgeMesh::New();

    // Map from grid linear index to HEM vertex index
    std::unordered_map<int, std::size_t> gridToVertex;
    // Map from HEM vertex index to grid linear index
    std::unordered_map<std::size_t, int> vertexToGrid;

    // First pass: collect all valid vertices from valid quads
    std::unordered_map<int, bool> usedVertices;
    for (int row = 0; row < points->rows - 1; ++row) {
        for (int col = 0; col < points->cols - 1; ++col) {
            const cv::Vec3f& p00 = (*points)(row, col);
            const cv::Vec3f& p01 = (*points)(row, col + 1);
            const cv::Vec3f& p10 = (*points)(row + 1, col);
            const cv::Vec3f& p11 = (*points)(row + 1, col + 1);

            if (p00[0] == -1.f || p01[0] == -1.f || p10[0] == -1.f || p11[0] == -1.f)
                continue;

            // Mark vertices as used
            usedVertices[row * points->cols + col] = true;
            usedVertices[row * points->cols + (col + 1)] = true;
            usedVertices[(row + 1) * points->cols + col] = true;
            usedVertices[(row + 1) * points->cols + (col + 1)] = true;
        }
    }

    if (usedVertices.empty()) {
        std::cerr << "ABF++: No valid quads found" << std::endl;
        return cv::Mat_<cv::Vec2f>();
    }

    // Step 1: Insert vertices
    std::size_t vertexIdx = 0;
    for (const auto& [linearIdx, _] : usedVertices) {
        int row = linearIdx / points->cols;
        int col = linearIdx % points->cols;
        const cv::Vec3f& pt = (*points)(row, col);

        OpenABF::Vec3d p;
        p[0] = pt[0];
        p[1] = pt[1];
        p[2] = pt[2];
        hem->insert_vertex(p);

        gridToVertex[linearIdx] = vertexIdx;
        vertexToGrid[vertexIdx] = linearIdx;
        vertexIdx++;
    }

    std::cout << "ABF++: Inserted " << vertexIdx << " vertices" << std::endl;

    // Step 2: Insert faces (triangulated quads)
    int faceCount = 0;
    for (int row = 0; row < points->rows - 1; ++row) {
        for (int col = 0; col < points->cols - 1; ++col) {
            const cv::Vec3f& p00 = (*points)(row, col);
            const cv::Vec3f& p01 = (*points)(row, col + 1);
            const cv::Vec3f& p10 = (*points)(row + 1, col);
            const cv::Vec3f& p11 = (*points)(row + 1, col + 1);

            if (p00[0] == -1.f || p01[0] == -1.f || p10[0] == -1.f || p11[0] == -1.f)
                continue;

            int idx00 = row * points->cols + col;
            int idx01 = row * points->cols + (col + 1);
            int idx10 = (row + 1) * points->cols + col;
            int idx11 = (row + 1) * points->cols + (col + 1);

            std::size_t v00 = gridToVertex[idx00];
            std::size_t v01 = gridToVertex[idx01];
            std::size_t v10 = gridToVertex[idx10];
            std::size_t v11 = gridToVertex[idx11];

            // Triangle 1: p10, p00, p01 (matching vc_tifxyz2obj winding)
            std::vector<std::size_t> face1 = {v10, v00, v01};
            hem->insert_face(face1);

            // Triangle 2: p10, p01, p11
            std::vector<std::size_t> face2 = {v10, v01, v11};
            hem->insert_face(face2);

            faceCount += 2;
        }
    }

    std::cout << "ABF++: Inserted " << faceCount << " faces" << std::endl;

    hem->update_boundary();

    // Step 3: Check manifold
    if (!OpenABF::IsManifold(hem)) {
        std::cerr << "ABF++: Mesh is not manifold" << std::endl;
        return cv::Mat_<cv::Vec2f>();
    }

    // Step 4: Run ABF++ optimization
    if (config.useABF) {
        std::cout << "ABF++: Running angle optimization (max " << config.maxIterations << " iterations)..." << std::endl;
        std::size_t iters = 0;
        double grad = 0;
        try {
            ABF::Compute(hem, iters, grad, config.maxIterations);
            std::cout << "ABF++: Completed in " << iters << " iterations, final grad: " << grad << std::endl;
        } catch (const OpenABF::SolverException& e) {
            std::cerr << "ABF++: Solver failed (" << e.what() << "), falling back to LSCM only" << std::endl;
        }
    }

    // Step 5: Run LSCM for final parameterization
    std::cout << "ABF++: Running LSCM parameterization..." << std::endl;
    try {
        LSCM::Compute(hem);
    } catch (const std::exception& e) {
        std::cerr << "ABF++: LSCM failed: " << e.what() << std::endl;
        return cv::Mat_<cv::Vec2f>();
    }

    // Step 6: Extract UVs and map back to grid
    for (const auto& v : hem->vertices()) {
        int linearIdx = vertexToGrid[v->idx];
        int row = linearIdx / points->cols;
        int col = linearIdx % points->cols;

        // OpenABF stores result in pos[0], pos[1]
        uvs(row, col) = cv::Vec2f(
            static_cast<float>(v->pos[0]),
            static_cast<float>(v->pos[1])
        );
    }

    // Step 7: Scale to match original surface area (optional)
    if (config.scaleToOriginalArea) {
        double area3D = computeSurfaceArea3D(surface);
        double area2D = computeArea2D(uvs, surface);

        if (area2D > 1e-10) {
            double scale = std::sqrt(area3D / area2D);
            std::cout << "ABF++: Scaling UVs by " << scale << " to match 3D area" << std::endl;

            for (int row = 0; row < uvs.rows; ++row) {
                for (int col = 0; col < uvs.cols; ++col) {
                    if (uvs(row, col)[0] != -1.f) {
                        uvs(row, col) *= static_cast<float>(scale);
                    }
                }
            }
        }
    }

    std::cout << "ABF++: Flattening complete" << std::endl;
    return uvs;
}

cv::Mat_<cv::Vec2f> abfFlatten(const QuadSurface& surface, const ABFConfig& config) {
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    if (!points || points->empty()) {
        std::cerr << "ABF++: Empty surface" << std::endl;
        return cv::Mat_<cv::Vec2f>();
    }

    // If no downsampling requested, run directly
    if (config.downsampleFactor <= 1) {
        return abfFlattenInternal(surface, config);
    }

    // Downsample the grid
    int originalRows = points->rows;
    int originalCols = points->cols;

    std::cout << "ABF++: Downsampling grid by factor " << config.downsampleFactor
              << " (" << originalRows << "x" << originalCols << " -> ";

    cv::Mat_<cv::Vec3f> coarseGrid = downsampleGrid(*points, config.downsampleFactor);

    std::cout << coarseGrid.rows << "x" << coarseGrid.cols << ")" << std::endl;

    // Create a temporary coarse surface for ABF
    // Note: We need to allocate this on the heap and manage it carefully
    cv::Mat_<cv::Vec3f>* coarsePointsPtr = new cv::Mat_<cv::Vec3f>(coarseGrid);
    cv::Vec2f coarseScale = surface._scale * static_cast<float>(config.downsampleFactor);
    QuadSurface coarseSurface(coarsePointsPtr, coarseScale);

    // Run ABF on coarse grid
    cv::Mat_<cv::Vec2f> coarseUVs = abfFlattenInternal(coarseSurface, config);

    if (coarseUVs.empty()) {
        return cv::Mat_<cv::Vec2f>();
    }

    // Upsample UVs back to original resolution
    std::cout << "ABF++: Upsampling UVs from " << coarseUVs.rows << "x" << coarseUVs.cols
              << " to " << originalRows << "x" << originalCols << std::endl;

    return upsampleUVs(coarseUVs, originalRows, originalCols, config.downsampleFactor);
}

bool abfFlattenInPlace(QuadSurface& surface, const ABFConfig& config) {
    cv::Mat_<cv::Vec2f> uvs = abfFlatten(surface, config);
    if (uvs.empty()) {
        return false;
    }

    surface.setChannel("uv", uvs);
    return true;
}

/**
 * @brief Precomputed triangle invariants for fast barycentric computation
 *
 * These values only depend on the triangle vertices and can be computed once
 * per triangle instead of once per pixel.
 */
struct TriangleInvariants {
    cv::Vec2f a;        // First vertex (reference point)
    cv::Vec2f v0, v1;   // Edge vectors: v0 = c - a, v1 = b - a
    float dot00, dot01, dot11;
    float invDenom;
    bool degenerate;
};

/**
 * @brief Precompute triangle invariants for fast barycentric testing
 */
static TriangleInvariants precomputeTriangle(const cv::Vec2f& a,
                                              const cv::Vec2f& b,
                                              const cv::Vec2f& c) {
    TriangleInvariants inv;
    inv.a = a;
    inv.v0 = c - a;
    inv.v1 = b - a;
    inv.dot00 = inv.v0.dot(inv.v0);
    inv.dot01 = inv.v0.dot(inv.v1);
    inv.dot11 = inv.v1.dot(inv.v1);
    float denom = inv.dot00 * inv.dot11 - inv.dot01 * inv.dot01;
    inv.degenerate = (std::fabs(denom) < 1e-20f || !std::isfinite(denom));
    inv.invDenom = inv.degenerate ? 0.0f : (1.0f / denom);
    return inv;
}

/**
 * @brief Fast barycentric computation using precomputed triangle invariants
 */
static cv::Vec3f computeBarycentricFast(const cv::Vec2f& p,
                                         const TriangleInvariants& inv) {
    if (inv.degenerate) {
        return cv::Vec3f(-1.f, -1.f, -1.f);
    }
    cv::Vec2f v2 = p - inv.a;
    float dot02 = inv.v0.dot(v2);
    float dot12 = inv.v1.dot(v2);
    float u = (inv.dot11 * dot02 - inv.dot01 * dot12) * inv.invDenom;
    float v = (inv.dot00 * dot12 - inv.dot01 * dot02) * inv.invDenom;
    return cv::Vec3f(1.0f - u - v, v, u);
}

QuadSurface* abfFlattenToNewSurface(const QuadSurface& surface, const ABFConfig& config) {
    // Step 1: Compute flattened UVs
    cv::Mat_<cv::Vec2f> uvs = abfFlatten(surface, config);
    if (uvs.empty()) {
        return nullptr;
    }

    const cv::Mat_<cv::Vec3f>* srcPoints = surface.rawPointsPtr();

    // Step 2: Find UV bounds (parallelized)
    float uvMinX = std::numeric_limits<float>::max();
    float uvMinY = std::numeric_limits<float>::max();
    float uvMaxX = std::numeric_limits<float>::lowest();
    float uvMaxY = std::numeric_limits<float>::lowest();

    #pragma omp parallel for collapse(2) reduction(min:uvMinX,uvMinY) reduction(max:uvMaxX,uvMaxY)
    for (int row = 0; row < uvs.rows; ++row) {
        for (int col = 0; col < uvs.cols; ++col) {
            const cv::Vec2f& uv = uvs(row, col);
            if (uv[0] != -1.f) {
                uvMinX = std::min(uvMinX, uv[0]);
                uvMinY = std::min(uvMinY, uv[1]);
                uvMaxX = std::max(uvMaxX, uv[0]);
                uvMaxY = std::max(uvMaxY, uv[1]);
            }
        }
    }
    cv::Vec2f uvMin(uvMinX, uvMinY);
    cv::Vec2f uvMax(uvMaxX, uvMaxY);

    cv::Vec2f uvRange = uvMax - uvMin;
    std::cout << "ABF++: UV bounds: [" << uvMin[0] << ", " << uvMin[1] << "] to ["
              << uvMax[0] << ", " << uvMax[1] << "]" << std::endl;

    // Step 3: Determine output grid size
    // The stored tifxyz grid is a downsampled representation. The scale factor
    // indicates how many voxels each grid cell represents (e.g., scale=0.05 means
    // 1 grid cell = 0.05 voxels, or equivalently, 20 grid cells per voxel).
    // When rendering, gen() upscales by 1/scale to get full resolution.
    //
    // UV coordinates after ABF++ (with scaleToOriginalArea=true) are in voxel units.
    // To get the output grid size, multiply UV range by input scale.
    float inputScaleX = surface._scale[0];
    float inputScaleY = surface._scale[1];

    int gridW = std::max(2, static_cast<int>(std::ceil(uvRange[0] * inputScaleX)) + 1);
    int gridH = std::max(2, static_cast<int>(std::ceil(uvRange[1] * inputScaleY)) + 1);

    std::cout << "ABF++: Creating output grid " << gridW << " x " << gridH
              << " (input scale=" << inputScaleX << "x" << inputScaleY
              << ", UV range=" << uvRange[0] << "x" << uvRange[1] << ")" << std::endl;

    // Step 4: Create output points grid
    cv::Mat_<cv::Vec3f>* outPoints = new cv::Mat_<cv::Vec3f>(gridH, gridW, cv::Vec3f(-1.f, -1.f, -1.f));

    // Step 5: Rasterize triangles onto the output grid (parallelized)
    // Precompute UV-to-grid transform factors
    const float rxInv = (gridW - 1) / std::max(uvRange[0], 1e-12f);
    const float ryInv = (gridH - 1) / std::max(uvRange[1], 1e-12f);

    // For each valid quad in the source, triangulate and rasterize
    #pragma omp parallel for collapse(2) schedule(dynamic, 64)
    for (int row = 0; row < srcPoints->rows - 1; ++row) {
        for (int col = 0; col < srcPoints->cols - 1; ++col) {
            const cv::Vec3f& p00 = (*srcPoints)(row, col);
            const cv::Vec3f& p01 = (*srcPoints)(row, col + 1);
            const cv::Vec3f& p10 = (*srcPoints)(row + 1, col);
            const cv::Vec3f& p11 = (*srcPoints)(row + 1, col + 1);

            if (p00[0] == -1.f || p01[0] == -1.f || p10[0] == -1.f || p11[0] == -1.f)
                continue;

            const cv::Vec2f& uv00 = uvs(row, col);
            const cv::Vec2f& uv01 = uvs(row, col + 1);
            const cv::Vec2f& uv10 = uvs(row + 1, col);
            const cv::Vec2f& uv11 = uvs(row + 1, col + 1);

            // Transform UVs to grid coordinates (inlined for performance)
            cv::Vec2f guv00((uv00[0] - uvMin[0]) * rxInv, (uv00[1] - uvMin[1]) * ryInv);
            cv::Vec2f guv01((uv01[0] - uvMin[0]) * rxInv, (uv01[1] - uvMin[1]) * ryInv);
            cv::Vec2f guv10((uv10[0] - uvMin[0]) * rxInv, (uv10[1] - uvMin[1]) * ryInv);
            cv::Vec2f guv11((uv11[0] - uvMin[0]) * rxInv, (uv11[1] - uvMin[1]) * ryInv);

            // Rasterize Triangle 1: (p10, p00, p01) with UVs (guv10, guv00, guv01)
            {
                // Precompute triangle invariants ONCE per triangle
                TriangleInvariants inv1 = precomputeTriangle(guv10, guv00, guv01);
                if (!inv1.degenerate) {
                    int minX = std::max(0, static_cast<int>(std::floor(std::min({guv10[0], guv00[0], guv01[0]}))) - 1);
                    int maxX = std::min(gridW - 1, static_cast<int>(std::ceil(std::max({guv10[0], guv00[0], guv01[0]}))) + 1);
                    int minY = std::max(0, static_cast<int>(std::floor(std::min({guv10[1], guv00[1], guv01[1]}))) - 1);
                    int maxY = std::min(gridH - 1, static_cast<int>(std::ceil(std::max({guv10[1], guv00[1], guv01[1]}))) + 1);

                    for (int y = minY; y <= maxY; ++y) {
                        for (int x = minX; x <= maxX; ++x) {
                            cv::Vec2f gridPt(static_cast<float>(x), static_cast<float>(y));
                            cv::Vec3f bary = computeBarycentricFast(gridPt, inv1);

                            if (bary[0] >= 0 && bary[1] >= 0 && bary[2] >= 0) {
                                cv::Vec3f pos = bary[0] * p10 + bary[1] * p00 + bary[2] * p01;
                                if ((*outPoints)(y, x)[0] == -1.f) {
                                    (*outPoints)(y, x) = pos;
                                }
                            }
                        }
                    }
                }
            }

            // Rasterize Triangle 2: (p10, p01, p11) with UVs (guv10, guv01, guv11)
            {
                // Precompute triangle invariants ONCE per triangle
                TriangleInvariants inv2 = precomputeTriangle(guv10, guv01, guv11);
                if (!inv2.degenerate) {
                    int minX = std::max(0, static_cast<int>(std::floor(std::min({guv10[0], guv01[0], guv11[0]}))) - 1);
                    int maxX = std::min(gridW - 1, static_cast<int>(std::ceil(std::max({guv10[0], guv01[0], guv11[0]}))) + 1);
                    int minY = std::max(0, static_cast<int>(std::floor(std::min({guv10[1], guv01[1], guv11[1]}))) - 1);
                    int maxY = std::min(gridH - 1, static_cast<int>(std::ceil(std::max({guv10[1], guv01[1], guv11[1]}))) + 1);

                    for (int y = minY; y <= maxY; ++y) {
                        for (int x = minX; x <= maxX; ++x) {
                            cv::Vec2f gridPt(static_cast<float>(x), static_cast<float>(y));
                            cv::Vec3f bary = computeBarycentricFast(gridPt, inv2);

                            if (bary[0] >= 0 && bary[1] >= 0 && bary[2] >= 0) {
                                cv::Vec3f pos = bary[0] * p10 + bary[1] * p01 + bary[2] * p11;
                                if ((*outPoints)(y, x)[0] == -1.f) {
                                    (*outPoints)(y, x) = pos;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Count valid points
    int validCount = 0;
    for (int y = 0; y < gridH; ++y) {
        for (int x = 0; x < gridW; ++x) {
            if ((*outPoints)(y, x)[0] != -1.f) {
                validCount++;
            }
        }
    }
    std::cout << "ABF++: Rasterized " << validCount << " / " << (gridW * gridH)
              << " points (" << (100.0f * validCount / (gridW * gridH)) << "%)" << std::endl;

    // Step 6: Use input scale for output
    // The scale determines how gen() upscales the grid. Using the same scale
    // as input ensures consistent rendering behavior.
    cv::Vec2f outScale = surface._scale;

    QuadSurface* result = new QuadSurface(outPoints, outScale);

    // Step 7: Optionally rotate to place highest Z values at top (row 0)
    if (config.rotateHighZToTop) {
        result->orientZUp();
    }

    return result;
}

} // namespace vc
