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
 * @brief Compute 3D surface area using triangulation of valid quads
 */
static double computeSurfaceArea3D(const QuadSurface& surface) {
    double area = 0.0;
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();

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
 * @brief Compute 2D area from UV coordinates
 */
static double computeArea2D(const cv::Mat_<cv::Vec2f>& uvs, const QuadSurface& surface) {
    double area = 0.0;
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();

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

cv::Mat_<cv::Vec2f> abfFlatten(const QuadSurface& surface, const ABFConfig& config) {
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

bool abfFlattenInPlace(QuadSurface& surface, const ABFConfig& config) {
    cv::Mat_<cv::Vec2f> uvs = abfFlatten(surface, config);
    if (uvs.empty()) {
        return false;
    }

    surface.setChannel("uv", uvs);
    return true;
}

/**
 * @brief Compute barycentric coordinates for point p in triangle (a, b, c)
 */
static cv::Vec3f computeBarycentric(const cv::Vec2f& p, const cv::Vec2f& a,
                                     const cv::Vec2f& b, const cv::Vec2f& c) {
    cv::Vec2f v0 = c - a;
    cv::Vec2f v1 = b - a;
    cv::Vec2f v2 = p - a;

    float dot00 = v0.dot(v0);
    float dot01 = v0.dot(v1);
    float dot02 = v0.dot(v2);
    float dot11 = v1.dot(v1);
    float dot12 = v1.dot(v2);

    const float denom = (dot00 * dot11 - dot01 * dot01);
    if (std::fabs(denom) < 1e-20f || !std::isfinite(denom)) {
        // Degenerate triangle in UV space
        return cv::Vec3f(-1.f, -1.f, -1.f);
    }
    float invDenom = 1.0f / denom;
    float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    return cv::Vec3f(1.0f - u - v, v, u);
}

QuadSurface* abfFlattenToNewSurface(const QuadSurface& surface, const ABFConfig& config) {
    // Step 1: Compute flattened UVs
    cv::Mat_<cv::Vec2f> uvs = abfFlatten(surface, config);
    if (uvs.empty()) {
        return nullptr;
    }

    const cv::Mat_<cv::Vec3f>* srcPoints = surface.rawPointsPtr();

    // Step 2: Find UV bounds
    cv::Vec2f uvMin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    cv::Vec2f uvMax(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());

    for (int row = 0; row < uvs.rows; ++row) {
        for (int col = 0; col < uvs.cols; ++col) {
            const cv::Vec2f& uv = uvs(row, col);
            if (uv[0] != -1.f) {
                uvMin[0] = std::min(uvMin[0], uv[0]);
                uvMin[1] = std::min(uvMin[1], uv[1]);
                uvMax[0] = std::max(uvMax[0], uv[0]);
                uvMax[1] = std::max(uvMax[1], uv[1]);
            }
        }
    }

    cv::Vec2f uvRange = uvMax - uvMin;
    std::cout << "ABF++: UV bounds: [" << uvMin[0] << ", " << uvMin[1] << "] to ["
              << uvMax[0] << ", " << uvMax[1] << "]" << std::endl;

    // Step 3: Determine output grid size
    // Use the original scale to maintain similar pixel density
    cv::Vec2f origScale = surface.scale();
    int gridW = std::max(2, static_cast<int>(std::ceil(uvRange[0] / origScale[0])) + 1);
    int gridH = std::max(2, static_cast<int>(std::ceil(uvRange[1] / origScale[1])) + 1);

    // Cap at reasonable size
    const int maxDim = 16384;
    if (gridW > maxDim || gridH > maxDim) {
        float downsample = std::max(
            static_cast<float>(gridW) / maxDim,
            static_cast<float>(gridH) / maxDim
        );
        gridW = std::max(2, static_cast<int>(gridW / downsample));
        gridH = std::max(2, static_cast<int>(gridH / downsample));
        std::cout << "ABF++: Capped output grid to " << gridW << " x " << gridH << std::endl;
    }

    std::cout << "ABF++: Creating output grid " << gridW << " x " << gridH << std::endl;

    // Step 4: Create output points grid
    cv::Mat_<cv::Vec3f>* outPoints = new cv::Mat_<cv::Vec3f>(gridH, gridW, cv::Vec3f(-1.f, -1.f, -1.f));

    // Step 5: Rasterize triangles onto the output grid
    // For each valid quad in the source, triangulate and rasterize
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

            // Transform UVs to grid coordinates
            auto uvToGrid = [&](const cv::Vec2f& uv) -> cv::Vec2f {
                float rx = std::max(uvRange[0], 1e-12f);
                float ry = std::max(uvRange[1], 1e-12f);
                return cv::Vec2f(
                    (uv[0] - uvMin[0]) / rx * (gridW - 1),
                    (uv[1] - uvMin[1]) / ry * (gridH - 1)
                );
            };

            cv::Vec2f guv00 = uvToGrid(uv00);
            cv::Vec2f guv01 = uvToGrid(uv01);
            cv::Vec2f guv10 = uvToGrid(uv10);
            cv::Vec2f guv11 = uvToGrid(uv11);

            // Rasterize Triangle 1: (p10, p00, p01) with UVs (guv10, guv00, guv01)
            {
                int minX = std::max(0, static_cast<int>(std::floor(std::min({guv10[0], guv00[0], guv01[0]}))) - 1);
                int maxX = std::min(gridW - 1, static_cast<int>(std::ceil(std::max({guv10[0], guv00[0], guv01[0]}))) + 1);
                int minY = std::max(0, static_cast<int>(std::floor(std::min({guv10[1], guv00[1], guv01[1]}))) - 1);
                int maxY = std::min(gridH - 1, static_cast<int>(std::ceil(std::max({guv10[1], guv00[1], guv01[1]}))) + 1);

                for (int y = minY; y <= maxY; ++y) {
                    for (int x = minX; x <= maxX; ++x) {
                        cv::Vec2f gridPt(static_cast<float>(x), static_cast<float>(y));
                        cv::Vec3f bary = computeBarycentric(gridPt, guv10, guv00, guv01);

                        if (bary[0] >= 0 && bary[1] >= 0 && bary[2] >= 0) {
                            cv::Vec3f pos = bary[0] * p10 + bary[1] * p00 + bary[2] * p01;
                            if ((*outPoints)(y, x)[0] == -1.f) {
                                (*outPoints)(y, x) = pos;
                            }
                        }
                    }
                }
            }

            // Rasterize Triangle 2: (p10, p01, p11) with UVs (guv10, guv01, guv11)
            {
                int minX = std::max(0, static_cast<int>(std::floor(std::min({guv10[0], guv01[0], guv11[0]}))) - 1);
                int maxX = std::min(gridW - 1, static_cast<int>(std::ceil(std::max({guv10[0], guv01[0], guv11[0]}))) + 1);
                int minY = std::max(0, static_cast<int>(std::floor(std::min({guv10[1], guv01[1], guv11[1]}))) - 1);
                int maxY = std::min(gridH - 1, static_cast<int>(std::ceil(std::max({guv10[1], guv01[1], guv11[1]}))) + 1);

                for (int y = minY; y <= maxY; ++y) {
                    for (int x = minX; x <= maxX; ++x) {
                        cv::Vec2f gridPt(static_cast<float>(x), static_cast<float>(y));
                        cv::Vec3f bary = computeBarycentric(gridPt, guv10, guv01, guv11);

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

    // Step 6: Compute output scale
    // The scale should match the UV coordinate spacing
    cv::Vec2f outScale(
        uvRange[0] / (gridW - 1),
        uvRange[1] / (gridH - 1)
    );

    return new QuadSurface(outPoints, outScale);
}

} // namespace vc
