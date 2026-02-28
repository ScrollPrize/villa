#include "vc/flattening/ABFFlattening.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <OpenABF/OpenABF.hpp>

#include <Eigen/Core>
#include <opencv2/imgproc.hpp>

#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <cmath>
#include <limits>
#include <array>
#include <vector>
#include <algorithm>
#include <cstdint>

namespace vc {

// Type aliases for OpenABF
using HalfEdgeMesh = OpenABF::detail::ABF::Mesh<double>;
using ABF = OpenABF::ABFPlusPlus<double>;
using LSCM = OpenABF::AngleBasedLSCM<double, HalfEdgeMesh>;

static inline bool isValidSurfacePoint(const cv::Vec3f& p)
{
    return p[0] != -1.f && std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

static inline bool hasZeroLengthEdge(const cv::Vec3f& a, const cv::Vec3f& b, const cv::Vec3f& c)
{
    const cv::Vec3f ab = b - a;
    const cv::Vec3f bc = c - b;
    const cv::Vec3f ca = a - c;
    return ab.dot(ab) == 0.f || bc.dot(bc) == 0.f || ca.dot(ca) == 0.f;
}

static inline bool isUsableTriangle(const cv::Vec3f& a, const cv::Vec3f& b, const cv::Vec3f& c)
{
    if (!isValidSurfacePoint(a) || !isValidSurfacePoint(b) || !isValidSurfacePoint(c)) {
        return false;
    }
    return !hasZeroLengthEdge(a, b, c);
}

static inline float triangleQuality(const cv::Vec3f& a, const cv::Vec3f& b, const cv::Vec3f& c)
{
    const cv::Vec3f ab = b - a;
    const cv::Vec3f bc = c - b;
    const cv::Vec3f ca = a - c;
    const float lab2 = ab.dot(ab);
    const float lbc2 = bc.dot(bc);
    const float lca2 = ca.dot(ca);
    const float perimeterSq = lab2 + lbc2 + lca2;
    if (!std::isfinite(perimeterSq) || perimeterSq <= 1e-20f) {
        return 0.f;
    }
    // Use 2*area (cross magnitude) to avoid extra constants in quality metric.
    const cv::Vec3f ac = c - a;
    const cv::Vec3f cross(
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0]
    );
    const float twoArea = std::sqrt(cross.dot(cross));
    if (!std::isfinite(twoArea) || twoArea <= 0.f) {
        return 0.f;
    }
    // Scale-invariant quality in [0, 1] for equilateral at 1.
    return (2.f * std::sqrt(3.f) * twoArea) / perimeterSq;
}

static inline bool isUsableTriangle(const cv::Vec3f& a, const cv::Vec3f& b, const cv::Vec3f& c,
                                    float minEdgeSq, float maxEdgeSq, float minQuality)
{
    if (!isValidSurfacePoint(a) || !isValidSurfacePoint(b) || !isValidSurfacePoint(c)) {
        return false;
    }
    const cv::Vec3f ab = b - a;
    const cv::Vec3f bc = c - b;
    const cv::Vec3f ca = a - c;
    const float lab2 = ab.dot(ab);
    const float lbc2 = bc.dot(bc);
    const float lca2 = ca.dot(ca);
    if (!std::isfinite(lab2) || !std::isfinite(lbc2) || !std::isfinite(lca2)) {
        return false;
    }
    if (lab2 <= minEdgeSq || lbc2 <= minEdgeSq || lca2 <= minEdgeSq) {
        return false;
    }
    if (lab2 >= maxEdgeSq || lbc2 >= maxEdgeSq || lca2 >= maxEdgeSq) {
        return false;
    }
    return triangleQuality(a, b, c) >= minQuality;
}

struct EdgeLengthStats {
    float p05 = 0.f;
    float p50 = 0.f;
    float p95 = 0.f;
    float p99 = 0.f;
    std::size_t count = 0;
};

static EdgeLengthStats computeGridEdgeLengthStats(const cv::Mat_<cv::Vec3f>& points)
{
    std::vector<float> lengths;
    lengths.reserve(static_cast<std::size_t>(points.rows * points.cols * 2));

    auto addEdge = [&](const cv::Vec3f& a, const cv::Vec3f& b) {
        if (!isValidSurfacePoint(a) || !isValidSurfacePoint(b)) {
            return;
        }
        const cv::Vec3f d = b - a;
        const float len = std::sqrt(d.dot(d));
        if (std::isfinite(len)) {
            lengths.push_back(len);
        }
    };

    for (int row = 0; row < points.rows; ++row) {
        for (int col = 0; col < points.cols; ++col) {
            if (col + 1 < points.cols) {
                addEdge(points(row, col), points(row, col + 1));
            }
            if (row + 1 < points.rows) {
                addEdge(points(row, col), points(row + 1, col));
            }
        }
    }

    EdgeLengthStats stats;
    stats.count = lengths.size();
    if (lengths.empty()) {
        return stats;
    }

    std::sort(lengths.begin(), lengths.end());
    auto quantile = [&](float q) -> float {
        q = std::clamp(q, 0.f, 1.f);
        const std::size_t idx = static_cast<std::size_t>(q * static_cast<float>(lengths.size() - 1));
        return lengths[idx];
    };

    stats.p05 = quantile(0.05f);
    stats.p50 = quantile(0.50f);
    stats.p95 = quantile(0.95f);
    stats.p99 = quantile(0.99f);
    return stats;
}

static void regularizeGridSpacing(cv::Mat_<cv::Vec3f>& grid, float targetSpacingVx)
{
    if (grid.empty() || targetSpacingVx <= 0.f) {
        return;
    }

    const int rows = grid.rows;
    const int cols = grid.cols;
    cv::Mat_<cv::Vec3f> anchor = grid.clone();

    constexpr int kIters = 24;
    constexpr float kStep = 0.35f;
    constexpr float kAnchor = 0.06f;
    const float maxEdgeCorrection = targetSpacingVx * 0.6f;

    auto isValidAt = [&](const cv::Mat_<cv::Vec3f>& g, int row, int col) -> bool {
        return isValidSurfacePoint(g(row, col));
    };

    for (int iter = 0; iter < kIters; ++iter) {
        cv::Mat_<cv::Vec3f> delta(rows, cols, cv::Vec3f(0.f, 0.f, 0.f));
        cv::Mat_<float> weights(rows, cols, 0.f);

        auto relaxEdge = [&](int r0, int c0, int r1, int c1) {
            if (!isValidAt(grid, r0, c0) || !isValidAt(grid, r1, c1)) {
                return;
            }

            const cv::Vec3f p0 = grid(r0, c0);
            const cv::Vec3f p1 = grid(r1, c1);
            const cv::Vec3f d = p0 - p1;
            const float lenSq = d.dot(d);
            if (!std::isfinite(lenSq) || lenSq <= 1e-10f) {
                return;
            }
            const float len = std::sqrt(lenSq);
            float err = len - targetSpacingVx;
            if (!std::isfinite(err)) {
                return;
            }
            err = std::clamp(err, -maxEdgeCorrection, maxEdgeCorrection);
            const cv::Vec3f dir = d * (1.f / len);
            const cv::Vec3f corr = 0.5f * err * dir;

            delta(r0, c0) += -corr;
            delta(r1, c1) += corr;
            weights(r0, c0) += 1.f;
            weights(r1, c1) += 1.f;
        };

        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                if (col + 1 < cols) {
                    relaxEdge(row, col, row, col + 1);
                }
                if (row + 1 < rows) {
                    relaxEdge(row, col, row + 1, col);
                }
            }
        }

        cv::Mat_<cv::Vec3f> next = grid.clone();
        int moved = 0;
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                if (!isValidAt(grid, row, col)) {
                    continue;
                }
                const float w = weights(row, col);
                if (w <= 0.f) {
                    continue;
                }
                const cv::Vec3f candidate = grid(row, col) + (kStep / w) * delta(row, col);
                if (!std::isfinite(candidate[0]) || !std::isfinite(candidate[1]) || !std::isfinite(candidate[2])) {
                    continue;
                }
                const cv::Vec3f anchored = (1.f - kAnchor) * candidate + kAnchor * anchor(row, col);
                next(row, col) = anchored;
                moved++;
            }
        }
        grid = std::move(next);
        if (moved == 0) {
            break;
        }
    }

    const EdgeLengthStats spacing = computeGridEdgeLengthStats(grid);
    if (spacing.count < 32 || spacing.p50 <= 1e-6f) {
        return;
    }

    const float globalScale = targetSpacingVx / spacing.p50;
    if (!std::isfinite(globalScale) || std::fabs(globalScale - 1.f) < 1e-4f) {
        return;
    }

    cv::Vec3f centroid(0.f, 0.f, 0.f);
    int validCount = 0;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            if (!isValidSurfacePoint(grid(row, col))) {
                continue;
            }
            centroid += grid(row, col);
            validCount++;
        }
    }
    if (validCount <= 0) {
        return;
    }
    centroid *= 1.f / static_cast<float>(validCount);

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            if (!isValidSurfacePoint(grid(row, col))) {
                continue;
            }
            const cv::Vec3f centered = grid(row, col) - centroid;
            grid(row, col) = centroid + globalScale * centered;
        }
    }
}

struct EdgeKey {
    int a;
    int b;

    bool operator==(const EdgeKey& other) const
    {
        return a == other.a && b == other.b;
    }
};

struct EdgeKeyHash {
    std::size_t operator()(const EdgeKey& k) const
    {
        const auto ua = static_cast<std::uint64_t>(static_cast<std::uint32_t>(k.a));
        const auto ub = static_cast<std::uint64_t>(static_cast<std::uint32_t>(k.b));
        return static_cast<std::size_t>((ua << 32) ^ ub);
    }
};

struct TriangleIdx {
    int i0;
    int i1;
    int i2;
};

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

static cv::Mat_<cv::Vec3f> regridPointsForFlattening(const cv::Mat_<cv::Vec3f>& points)
{
    cv::Mat_<cv::Vec3f> clone = points.clone();
    const int rows = clone.rows;
    const int cols = clone.cols;
    const int vertexCount = rows * cols;

    const auto linearIdx = [cols](int row, int col) -> int {
        return row * cols + col;
    };

    auto collectEdgeStats = [&](const cv::Mat_<cv::Vec3f>& grid,
                                std::vector<float>* outLengths,
                                std::vector<float>* outIncidentSum,
                                std::vector<int>* outIncidentCount,
                                std::vector<int>* outDegree) {
        outLengths->clear();
        outIncidentSum->assign(vertexCount, 0.f);
        outIncidentCount->assign(vertexCount, 0);
        outDegree->assign(vertexCount, 0);
        outLengths->reserve(static_cast<std::size_t>(rows * cols * 2));

        auto addEdge = [&](int r0, int c0, int r1, int c1) {
            const cv::Vec3f& a = grid(r0, c0);
            const cv::Vec3f& b = grid(r1, c1);
            if (!isValidSurfacePoint(a) || !isValidSurfacePoint(b)) {
                return;
            }
            const cv::Vec3f d = b - a;
            const float len = std::sqrt(d.dot(d));
            if (!std::isfinite(len)) {
                return;
            }
            outLengths->push_back(len);
            const int ia = linearIdx(r0, c0);
            const int ib = linearIdx(r1, c1);
            (*outIncidentSum)[ia] += len;
            (*outIncidentSum)[ib] += len;
            (*outIncidentCount)[ia] += 1;
            (*outIncidentCount)[ib] += 1;
            (*outDegree)[ia] += 1;
            (*outDegree)[ib] += 1;
        };

        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                if (col + 1 < cols) {
                    addEdge(row, col, row, col + 1);
                }
                if (row + 1 < rows) {
                    addEdge(row, col, row + 1, col);
                }
            }
        }
    };

    std::vector<float> edgeLengths;
    std::vector<float> incidentSum;
    std::vector<int> incidentCount;
    std::vector<int> degree;
    collectEdgeStats(clone, &edgeLengths, &incidentSum, &incidentCount, &degree);

    const float dedupeDistance = 5.f; // requested: merge/remove near-duplicates within ~5 voxels
    int removedOutlierOrDup = 0;
    int filledInvalid = 0;
    if (edgeLengths.size() >= 128) {
        std::sort(edgeLengths.begin(), edgeLengths.end());
        const auto quantile = [&](float q) -> float {
            q = std::clamp(q, 0.f, 1.f);
            const std::size_t idx = static_cast<std::size_t>(q * static_cast<float>(edgeLengths.size() - 1));
            return edgeLengths[idx];
        };

        const float p50 = quantile(0.50f);
        const float p95 = quantile(0.95f);
        const float shortEdgeThreshold = std::max(1e-3f, std::min(dedupeDistance, p50 * 0.30f));
        const float longEdgeThreshold = std::max({p95 * 3.0f, p50 * 6.0f, dedupeDistance * 4.0f});

        std::vector<int> removeVotes(vertexCount, 0);
        std::vector<int> shortVotes(vertexCount, 0);
        std::vector<int> longVotes(vertexCount, 0);

        const auto avgIncidentLen = [&](int idx) -> float {
            if (incidentCount[idx] <= 0) {
                return std::numeric_limits<float>::infinity();
            }
            return incidentSum[idx] / static_cast<float>(incidentCount[idx]);
        };

        auto voteEdge = [&](int r0, int c0, int r1, int c1) {
            const cv::Vec3f& a = clone(r0, c0);
            const cv::Vec3f& b = clone(r1, c1);
            if (!isValidSurfacePoint(a) || !isValidSurfacePoint(b)) {
                return;
            }
            const cv::Vec3f d = b - a;
            const float len = std::sqrt(d.dot(d));
            if (!std::isfinite(len)) {
                return;
            }

            const int ia = linearIdx(r0, c0);
            const int ib = linearIdx(r1, c1);

            if (len <= shortEdgeThreshold) {
                // Treat near-zero/very-short edges as duplicate weld candidates.
                int pick = ia;
                if (degree[ib] < degree[ia]) {
                    pick = ib;
                } else if (degree[ib] == degree[ia]) {
                    const float aAvg = avgIncidentLen(ia);
                    const float bAvg = avgIncidentLen(ib);
                    if (bAvg < aAvg) {
                        pick = ib;
                    }
                }
                removeVotes[pick] += 2;
                shortVotes[pick] += 1;
                return;
            }

            if (len >= longEdgeThreshold) {
                // Long incident edges indicate outlier vertices that bridge distant geometry.
                int pick = ia;
                const float aAvg = avgIncidentLen(ia);
                const float bAvg = avgIncidentLen(ib);
                if (std::isfinite(aAvg) && std::isfinite(bAvg) && std::fabs(aAvg - bAvg) > 1e-3f) {
                    pick = (aAvg > bAvg) ? ia : ib;
                } else if (degree[ib] < degree[ia]) {
                    pick = ib;
                }
                removeVotes[pick] += 1;
                longVotes[pick] += 1;
            }
        };

        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                if (col + 1 < cols) {
                    voteEdge(row, col, row, col + 1);
                }
                if (row + 1 < rows) {
                    voteEdge(row, col, row + 1, col);
                }
            }
        }

        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                if (!isValidSurfacePoint(clone(row, col))) {
                    continue;
                }
                const int idx = linearIdx(row, col);
                bool drop = false;
                if (shortVotes[idx] > 0) {
                    drop = true;
                } else if (removeVotes[idx] >= 2) {
                    drop = true;
                } else if (removeVotes[idx] >= 1 && degree[idx] <= 2) {
                    drop = true;
                }
                if (drop) {
                    clone(row, col) = cv::Vec3f(-1.f, -1.f, -1.f);
                    removedOutlierOrDup++;
                }
            }
        }

        if (removedOutlierOrDup > 0) {
            // Inpaint small holes from local neighbors before regridding.
            for (int iter = 0; iter < 8; ++iter) {
                cv::Mat_<cv::Vec3f> next = clone.clone();
                int changed = 0;
                for (int row = 0; row < rows; ++row) {
                    for (int col = 0; col < cols; ++col) {
                        if (isValidSurfacePoint(clone(row, col))) {
                            continue;
                        }
                        cv::Vec3f sum(0.f, 0.f, 0.f);
                        int neighbors = 0;
                        if (row > 0 && isValidSurfacePoint(clone(row - 1, col))) {
                            sum += clone(row - 1, col);
                            neighbors++;
                        }
                        if (row + 1 < rows && isValidSurfacePoint(clone(row + 1, col))) {
                            sum += clone(row + 1, col);
                            neighbors++;
                        }
                        if (col > 0 && isValidSurfacePoint(clone(row, col - 1))) {
                            sum += clone(row, col - 1);
                            neighbors++;
                        }
                        if (col + 1 < cols && isValidSurfacePoint(clone(row, col + 1))) {
                            sum += clone(row, col + 1);
                            neighbors++;
                        }
                        if (neighbors >= 3) {
                            next(row, col) = sum * (1.f / static_cast<float>(neighbors));
                            changed++;
                        }
                    }
                }
                clone = std::move(next);
                filledInvalid += changed;
                if (changed == 0) {
                    break;
                }
            }
        }

        if (removedOutlierOrDup > 0) {
            std::cout << "ABF++: Regrid pre-clean removed " << removedOutlierOrDup
                      << " outlier/duplicate vertices (short<=" << shortEdgeThreshold
                      << ", long>=" << longEdgeThreshold
                      << ", duplicate radius=" << dedupeDistance << " voxels)"
                      << std::endl;
            if (filledInvalid > 0) {
                std::cout << "ABF++: Regrid pre-clean inpainted " << filledInvalid
                          << " vertex slot(s) from local neighbors" << std::endl;
            }
        }
    }

    QuadSurface tmp(clone, cv::Vec2f(1.f, 1.f));

    // Regrid by down/up sampling to smooth local degeneracies while preserving
    // the global shape. This uses the same interpolation/mask cleanup path as
    // QuadSurface transforms.
    tmp.resample(0.5f, cv::INTER_LINEAR);
    tmp.resample(2.0f, cv::INTER_LINEAR);

    cv::Mat_<cv::Vec3f> reg = tmp.rawPoints();
    if (reg.rows != points.rows || reg.cols != points.cols) {
        cv::Mat resized;
        cv::resize(reg, resized, points.size(), 0, 0, cv::INTER_LINEAR);
        reg = resized;
    }

    auto countGridNeighbors = [&](int row, int col) -> int {
        int neighbors = 0;
        if (row > 0 && isValidSurfacePoint(reg(row - 1, col))) {
            neighbors++;
        }
        if (row + 1 < reg.rows && isValidSurfacePoint(reg(row + 1, col))) {
            neighbors++;
        }
        if (col > 0 && isValidSurfacePoint(reg(row, col - 1))) {
            neighbors++;
        }
        if (col + 1 < reg.cols && isValidSurfacePoint(reg(row, col + 1))) {
            neighbors++;
        }
        return neighbors;
    };

    int trimmedTotal = 0;
    for (int iter = 0; iter < 16; ++iter) {
        cv::Mat_<uint8_t> trimMask(reg.rows, reg.cols, uint8_t(0));
        int trimmedThisIter = 0;
        for (int row = 0; row < reg.rows; ++row) {
            for (int col = 0; col < reg.cols; ++col) {
                if (!isValidSurfacePoint(reg(row, col))) {
                    continue;
                }
                // Trim leaf vertices introduced by interpolation/mask edges.
                if (countGridNeighbors(row, col) <= 1) {
                    trimMask(row, col) = 1;
                    trimmedThisIter++;
                }
            }
        }
        if (trimmedThisIter == 0) {
            break;
        }
        for (int row = 0; row < reg.rows; ++row) {
            for (int col = 0; col < reg.cols; ++col) {
                if (trimMask(row, col) != 0) {
                    reg(row, col) = cv::Vec3f(-1.f, -1.f, -1.f);
                }
            }
        }
        trimmedTotal += trimmedThisIter;
    }
    if (trimmedTotal > 0) {
        std::cout << "ABF++: Regrid cleanup trimmed " << trimmedTotal
                  << " dangling edge vertices" << std::endl;
    }

    const EdgeLengthStats preSpacing = computeGridEdgeLengthStats(reg);
    const float targetSpacingVx = 20.f;
    regularizeGridSpacing(reg, targetSpacingVx);
    const EdgeLengthStats postSpacing = computeGridEdgeLengthStats(reg);
    if (preSpacing.count >= 128 && postSpacing.count >= 128) {
        std::cout << "ABF++: Grid spacing regularized to ~" << targetSpacingVx
                  << " vx (p05/p50/p95/p99: "
                  << preSpacing.p05 << "/" << preSpacing.p50 << "/" << preSpacing.p95 << "/" << preSpacing.p99
                  << " -> "
                  << postSpacing.p05 << "/" << postSpacing.p50 << "/" << postSpacing.p95 << "/" << postSpacing.p99
                  << ")" << std::endl;
    }

    return reg;
}

static void alignUVsToInputGrid(const cv::Mat_<cv::Vec3f>& points, cv::Mat_<cv::Vec2f>& uvs)
{
    auto isValidPoint = [&](int row, int col) -> bool {
        const cv::Vec3f& p = points(row, col);
        return p[0] != -1.f && std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
    };
    auto isValidUV = [&](int row, int col) -> bool {
        const cv::Vec2f& uv = uvs(row, col);
        return uv[0] != -1.f && std::isfinite(uv[0]) && std::isfinite(uv[1]);
    };

    cv::Vec2d dCol(0.0, 0.0);
    cv::Vec2d dRow(0.0, 0.0);
    std::size_t colSamples = 0;
    std::size_t rowSamples = 0;

    for (int row = 0; row < points.rows; ++row) {
        for (int col = 0; col + 1 < points.cols; ++col) {
            if (!isValidPoint(row, col) || !isValidPoint(row, col + 1) ||
                !isValidUV(row, col) || !isValidUV(row, col + 1)) {
                continue;
            }
            cv::Vec2f delta = uvs(row, col + 1) - uvs(row, col);
            dCol[0] += delta[0];
            dCol[1] += delta[1];
            ++colSamples;
        }
    }

    for (int row = 0; row + 1 < points.rows; ++row) {
        for (int col = 0; col < points.cols; ++col) {
            if (!isValidPoint(row, col) || !isValidPoint(row + 1, col) ||
                !isValidUV(row, col) || !isValidUV(row + 1, col)) {
                continue;
            }
            cv::Vec2f delta = uvs(row + 1, col) - uvs(row, col);
            dRow[0] += delta[0];
            dRow[1] += delta[1];
            ++rowSamples;
        }
    }

    if (colSamples == 0 && rowSamples == 0) {
        return;
    }

    const double colNorm = std::sqrt(dCol.dot(dCol));
    const double rowNorm = std::sqrt(dRow.dot(dRow));
    if (colNorm < 1e-12 && rowNorm < 1e-12) {
        return;
    }

    struct Mat2 {
        float a, b, c, d;
    };
    // D4: all axis-aligned rotations/reflections.
    const std::array<Mat2, 8> candidates{{
        { 1.f,  0.f,  0.f,  1.f}, // identity
        { 0.f, -1.f,  1.f,  0.f}, // rotate +90
        {-1.f,  0.f,  0.f, -1.f}, // rotate 180
        { 0.f,  1.f, -1.f,  0.f}, // rotate -90
        {-1.f,  0.f,  0.f,  1.f}, // mirror X
        { 1.f,  0.f,  0.f, -1.f}, // mirror Y
        { 0.f,  1.f,  1.f,  0.f}, // mirror y=x
        { 0.f, -1.f, -1.f,  0.f}  // mirror y=-x
    }};

    auto transform = [](const Mat2& m, const cv::Vec2d& v) -> cv::Vec2d {
        return cv::Vec2d(m.a * v[0] + m.b * v[1], m.c * v[0] + m.d * v[1]);
    };

    double bestScore = -std::numeric_limits<double>::infinity();
    Mat2 best = candidates.front();
    for (const auto& m : candidates) {
        double score = 0.0;
        if (colNorm >= 1e-12) {
            cv::Vec2d t = transform(m, dCol);
            score += t[0] / colNorm;               // prefer +U along +col
            score -= 0.25 * std::abs(t[1]) / colNorm; // penalize cross-axis
        }
        if (rowNorm >= 1e-12) {
            cv::Vec2d t = transform(m, dRow);
            score += t[1] / rowNorm;               // prefer +V along +row
            score -= 0.25 * std::abs(t[0]) / rowNorm; // penalize cross-axis
        }
        if (score > bestScore) {
            bestScore = score;
            best = m;
        }
    }

    for (int row = 0; row < uvs.rows; ++row) {
        for (int col = 0; col < uvs.cols; ++col) {
            if (!isValidUV(row, col)) {
                continue;
            }
            const cv::Vec2f uv = uvs(row, col);
            uvs(row, col) = cv::Vec2f(
                best.a * uv[0] + best.b * uv[1],
                best.c * uv[0] + best.d * uv[1]
            );
        }
    }
}

/**
 * @brief Internal ABF++ flattening on the provided surface (no downsampling)
 */
static cv::Mat_<cv::Vec2f> abfFlattenInternal(const QuadSurface& surface,
                                              const ABFConfig& config,
                                              int* skippedDegenerateOut = nullptr) {
    if (skippedDegenerateOut) {
        *skippedDegenerateOut = 0;
    }
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    if (!points || points->empty()) {
        std::cerr << "ABF++: Empty surface" << std::endl;
        return cv::Mat_<cv::Vec2f>();
    }

    // Initialize UV output with invalid values
    cv::Mat_<cv::Vec2f> uvs(points->size(), cv::Vec2f(-1.f, -1.f));

    // Build half-edge mesh from valid triangles.
    auto hem = HalfEdgeMesh::New();

    // Map from HEM vertex index to grid linear index
    std::unordered_map<std::size_t, int> vertexToGrid;

    // Step 1: Build candidate triangle list from valid quads.
    std::vector<TriangleIdx> triangles;
    triangles.reserve(static_cast<std::size_t>((points->rows - 1) * (points->cols - 1) * 2));
    int skippedHardDegenerate = 0;
    int skippedLowQuality = 0;

    const EdgeLengthStats edgeStats = computeGridEdgeLengthStats(*points);
    float minUsableEdge = 1e-6f;
    float maxUsableEdge = std::numeric_limits<float>::infinity();
    float minTriangleQuality = 1e-6f;
    if (edgeStats.count >= 128 && edgeStats.p50 > 0.f) {
        // Treat near-duplicate edges as degenerate and drop extreme long-edge bridges.
        minUsableEdge = std::max(1e-3f, std::min(5.f, edgeStats.p50 * 0.30f));
        maxUsableEdge = std::max(edgeStats.p95 * 3.0f, edgeStats.p50 * 6.0f);
        minTriangleQuality = 0.01f;
        std::cout << "ABF++: Triangle quality filter (minEdge=" << minUsableEdge
                  << ", maxEdge=" << maxUsableEdge
                  << ", minQuality=" << minTriangleQuality << ")" << std::endl;
    }
    const float minUsableEdgeSq = minUsableEdge * minUsableEdge;
    const float maxUsableEdgeSq = maxUsableEdge * maxUsableEdge;

    for (int row = 0; row < points->rows - 1; ++row) {
        for (int col = 0; col < points->cols - 1; ++col) {
            const cv::Vec3f& p00 = (*points)(row, col);
            const cv::Vec3f& p01 = (*points)(row, col + 1);
            const cv::Vec3f& p10 = (*points)(row + 1, col);
            const cv::Vec3f& p11 = (*points)(row + 1, col + 1);

            if (!isValidSurfacePoint(p00) || !isValidSurfacePoint(p01) ||
                !isValidSurfacePoint(p10) || !isValidSurfacePoint(p11)) {
                continue;
            }

            const int idx00 = row * points->cols + col;
            const int idx01 = row * points->cols + (col + 1);
            const int idx10 = (row + 1) * points->cols + col;
            const int idx11 = (row + 1) * points->cols + (col + 1);

            // Triangle 1: p10, p00, p01
            const bool tri1HardValid = isUsableTriangle(p10, p00, p01);
            const bool tri1Usable = tri1HardValid &&
                                    isUsableTriangle(p10, p00, p01, minUsableEdgeSq, maxUsableEdgeSq, minTriangleQuality);
            if (tri1Usable) {
                triangles.push_back({idx10, idx00, idx01});
            } else {
                if (tri1HardValid) {
                    skippedLowQuality++;
                } else {
                    skippedHardDegenerate++;
                }
            }

            // Triangle 2: p10, p01, p11
            const bool tri2HardValid = isUsableTriangle(p10, p01, p11);
            const bool tri2Usable = tri2HardValid &&
                                    isUsableTriangle(p10, p01, p11, minUsableEdgeSq, maxUsableEdgeSq, minTriangleQuality);
            if (tri2Usable) {
                triangles.push_back({idx10, idx01, idx11});
            } else {
                if (tri2HardValid) {
                    skippedLowQuality++;
                } else {
                    skippedHardDegenerate++;
                }
            }
        }
    }

    if (triangles.empty()) {
        std::cerr << "ABF++: No non-degenerate faces found" << std::endl;
        return cv::Mat_<cv::Vec2f>();
    }

    // Step 2: Build triangle connectivity by shared edges.
    std::unordered_map<EdgeKey, std::vector<int>, EdgeKeyHash> edgeToTriangles;
    edgeToTriangles.reserve(triangles.size() * 2);
    for (int triIdx = 0; triIdx < static_cast<int>(triangles.size()); ++triIdx) {
        const auto& t = triangles[triIdx];
        const EdgeKey e0{std::min(t.i0, t.i1), std::max(t.i0, t.i1)};
        const EdgeKey e1{std::min(t.i1, t.i2), std::max(t.i1, t.i2)};
        const EdgeKey e2{std::min(t.i2, t.i0), std::max(t.i2, t.i0)};
        edgeToTriangles[e0].push_back(triIdx);
        edgeToTriangles[e1].push_back(triIdx);
        edgeToTriangles[e2].push_back(triIdx);
    }

    std::vector<std::vector<int>> triAdj(triangles.size());
    for (const auto& [_, triList] : edgeToTriangles) {
        for (int i = 1; i < static_cast<int>(triList.size()); ++i) {
            triAdj[triList[0]].push_back(triList[i]);
            triAdj[triList[i]].push_back(triList[0]);
        }
    }

    // Step 3: Label edge-connected triangle components.
    std::vector<int> triComponent(triangles.size(), -1);
    int componentCount = 0;
    std::vector<int> stack;
    for (int triIdx = 0; triIdx < static_cast<int>(triangles.size()); ++triIdx) {
        if (triComponent[triIdx] != -1) {
            continue;
        }
        stack.clear();
        stack.push_back(triIdx);
        triComponent[triIdx] = componentCount;
        while (!stack.empty()) {
            const int cur = stack.back();
            stack.pop_back();
            for (int nb : triAdj[cur]) {
                if (triComponent[nb] == -1) {
                    triComponent[nb] = componentCount;
                    stack.push_back(nb);
                }
            }
        }
        componentCount++;
    }

    std::vector<int> componentTriCounts(componentCount, 0);
    for (int cid : triComponent) {
        componentTriCounts[cid]++;
    }
    int selectedComponent = 0;
    for (int cid = 1; cid < componentCount; ++cid) {
        if (componentTriCounts[cid] > componentTriCounts[selectedComponent]) {
            selectedComponent = cid;
        }
    }

    int droppedDisconnected = 0;
    std::vector<TriangleIdx> selectedTriangles;
    selectedTriangles.reserve(componentTriCounts[selectedComponent]);
    for (int triIdx = 0; triIdx < static_cast<int>(triangles.size()); ++triIdx) {
        const auto& t = triangles[triIdx];
        if (triComponent[triIdx] != selectedComponent) {
            droppedDisconnected++;
            continue;
        }
        selectedTriangles.push_back(t);
    }

    // Step 4: Resolve boundary pinch vertices where multiple boundary chains
    // meet at a single vertex (in/out degree != 1). Instead of deleting
    // triangles, split the pinch vertex into multiple copies, one per local
    // fan, preserving geometry and winding.
    const int gridVertexCount = points->rows * points->cols;
    int nextSyntheticVertex = gridVertexCount;
    std::unordered_map<int, int> syntheticToBaseGrid;
    int splitPinchVertices = 0;
    for (int iter = 0; iter < 8 && !selectedTriangles.empty(); ++iter) {
        std::unordered_map<EdgeKey, int, EdgeKeyHash> undirectedCounts;
        std::unordered_map<EdgeKey, int, EdgeKeyHash> directedCounts;
        undirectedCounts.reserve(selectedTriangles.size() * 2);
        directedCounts.reserve(selectedTriangles.size() * 2);
        for (const auto& t : selectedTriangles) {
            const std::array<EdgeKey, 3> undirected{
                EdgeKey{std::min(t.i0, t.i1), std::max(t.i0, t.i1)},
                EdgeKey{std::min(t.i1, t.i2), std::max(t.i1, t.i2)},
                EdgeKey{std::min(t.i2, t.i0), std::max(t.i2, t.i0)}
            };
            const std::array<EdgeKey, 3> directed{
                EdgeKey{t.i0, t.i1},
                EdgeKey{t.i1, t.i2},
                EdgeKey{t.i2, t.i0}
            };
            for (const auto& e : undirected) {
                ++undirectedCounts[e];
            }
            for (const auto& e : directed) {
                ++directedCounts[e];
            }
        }

        std::unordered_map<int, int> boundaryInDegree;
        std::unordered_map<int, int> boundaryOutDegree;
        boundaryInDegree.reserve(undirectedCounts.size());
        boundaryOutDegree.reserve(undirectedCounts.size());
        for (const auto& [edge, count] : undirectedCounts) {
            if (count != 1) {
                continue;
            }
            const int a = edge.a;
            const int b = edge.b;
            const int ab = directedCounts[EdgeKey{a, b}];
            const int ba = directedCounts[EdgeKey{b, a}];
            if (ab == 1 && ba == 0) {
                ++boundaryOutDegree[a];
                ++boundaryInDegree[b];
            } else if (ba == 1 && ab == 0) {
                ++boundaryOutDegree[b];
                ++boundaryInDegree[a];
            } else {
                ++boundaryOutDegree[a];
                ++boundaryInDegree[b];
            }
        }

        std::unordered_set<int> badBoundaryVerts;
        badBoundaryVerts.reserve(boundaryOutDegree.size() + boundaryInDegree.size());
        for (const auto& [v, out] : boundaryOutDegree) {
            const int in = boundaryInDegree[v];
            if (in != 1 || out != 1) {
                badBoundaryVerts.insert(v);
            }
        }
        for (const auto& [v, in] : boundaryInDegree) {
            if (boundaryOutDegree.find(v) != boundaryOutDegree.end()) {
                continue;
            }
            if (in != 1) {
                badBoundaryVerts.insert(v);
            } else {
                badBoundaryVerts.insert(v);
            }
        }

        if (badBoundaryVerts.empty()) {
            break;
        }

        bool changed = false;
        const auto baseGridIndex = [&](int vid) -> int {
            auto it = syntheticToBaseGrid.find(vid);
            return it == syntheticToBaseGrid.end() ? vid : it->second;
        };
        const auto triHasVertex = [](const TriangleIdx& t, int v) -> bool {
            return t.i0 == v || t.i1 == v || t.i2 == v;
        };
        const auto replaceTriVertex = [](TriangleIdx& t, int fromV, int toV) {
            if (t.i0 == fromV) t.i0 = toV;
            if (t.i1 == fromV) t.i1 = toV;
            if (t.i2 == fromV) t.i2 = toV;
        };

        for (int v : badBoundaryVerts) {
            std::vector<int> incident;
            incident.reserve(16);
            for (int triIdx = 0; triIdx < static_cast<int>(selectedTriangles.size()); ++triIdx) {
                if (triHasVertex(selectedTriangles[triIdx], v)) {
                    incident.push_back(triIdx);
                }
            }
            if (incident.size() < 2) {
                continue;
            }

            // Build local adjacency of incident triangles via shared edges that
            // include the pinch vertex.
            std::unordered_map<int, std::vector<int>> edgeToIncident;
            edgeToIncident.reserve(incident.size() * 2);
            for (int triIdx : incident) {
                const auto& t = selectedTriangles[triIdx];
                const std::array<int, 2> others = t.i0 == v ? std::array<int, 2>{t.i1, t.i2}
                                          : (t.i1 == v ? std::array<int, 2>{t.i0, t.i2}
                                                       : std::array<int, 2>{t.i0, t.i1});
                edgeToIncident[others[0]].push_back(triIdx);
                edgeToIncident[others[1]].push_back(triIdx);
            }

            std::unordered_map<int, std::vector<int>> triAdjLocal;
            triAdjLocal.reserve(incident.size());
            for (const auto& [_, trisForEdge] : edgeToIncident) {
                for (int i = 1; i < static_cast<int>(trisForEdge.size()); ++i) {
                    triAdjLocal[trisForEdge[0]].push_back(trisForEdge[i]);
                    triAdjLocal[trisForEdge[i]].push_back(trisForEdge[0]);
                }
            }

            std::unordered_set<int> seen;
            seen.reserve(incident.size());
            std::vector<std::vector<int>> localFans;
            for (int startTri : incident) {
                if (seen.count(startTri) > 0) {
                    continue;
                }
                std::vector<int> fan;
                std::vector<int> localStack{startTri};
                seen.insert(startTri);
                while (!localStack.empty()) {
                    const int curTri = localStack.back();
                    localStack.pop_back();
                    fan.push_back(curTri);
                    auto itAdj = triAdjLocal.find(curTri);
                    if (itAdj == triAdjLocal.end()) {
                        continue;
                    }
                    for (int nb : itAdj->second) {
                        if (seen.insert(nb).second) {
                            localStack.push_back(nb);
                        }
                    }
                }
                localFans.push_back(std::move(fan));
            }

            if (localFans.size() <= 1) {
                continue;
            }

            for (std::size_t fanIdx = 1; fanIdx < localFans.size(); ++fanIdx) {
                const int newVid = nextSyntheticVertex++;
                syntheticToBaseGrid[newVid] = baseGridIndex(v);
                for (int triIdx : localFans[fanIdx]) {
                    replaceTriVertex(selectedTriangles[triIdx], v, newVid);
                }
                splitPinchVertices++;
                changed = true;
            }
        }

        if (!changed) {
            break;
        }
    }

    if (selectedTriangles.empty()) {
        std::cerr << "ABF++: No valid triangles left after boundary cleanup" << std::endl;
        return cv::Mat_<cv::Vec2f>();
    }

    // Step 5: Cleanup can split the mesh; keep the largest edge-connected part.
    {
        std::unordered_map<EdgeKey, std::vector<int>, EdgeKeyHash> localEdgeToTriangles;
        localEdgeToTriangles.reserve(selectedTriangles.size() * 2);
        for (int triIdx = 0; triIdx < static_cast<int>(selectedTriangles.size()); ++triIdx) {
            const auto& t = selectedTriangles[triIdx];
            const EdgeKey e0{std::min(t.i0, t.i1), std::max(t.i0, t.i1)};
            const EdgeKey e1{std::min(t.i1, t.i2), std::max(t.i1, t.i2)};
            const EdgeKey e2{std::min(t.i2, t.i0), std::max(t.i2, t.i0)};
            localEdgeToTriangles[e0].push_back(triIdx);
            localEdgeToTriangles[e1].push_back(triIdx);
            localEdgeToTriangles[e2].push_back(triIdx);
        }

        std::vector<std::vector<int>> localAdj(selectedTriangles.size());
        for (const auto& [_, triList] : localEdgeToTriangles) {
            for (int i = 1; i < static_cast<int>(triList.size()); ++i) {
                localAdj[triList[0]].push_back(triList[i]);
                localAdj[triList[i]].push_back(triList[0]);
            }
        }

        std::vector<int> localComp(selectedTriangles.size(), -1);
        std::vector<int> localCompSize;
        int localCompCount = 0;
        for (int triIdx = 0; triIdx < static_cast<int>(selectedTriangles.size()); ++triIdx) {
            if (localComp[triIdx] != -1) {
                continue;
            }
            int compSize = 0;
            stack.clear();
            stack.push_back(triIdx);
            localComp[triIdx] = localCompCount;
            while (!stack.empty()) {
                const int cur = stack.back();
                stack.pop_back();
                compSize++;
                for (int nb : localAdj[cur]) {
                    if (localComp[nb] == -1) {
                        localComp[nb] = localCompCount;
                        stack.push_back(nb);
                    }
                }
            }
            localCompSize.push_back(compSize);
            localCompCount++;
        }

        int keepComp = 0;
        for (int cid = 1; cid < localCompCount; ++cid) {
            if (localCompSize[cid] > localCompSize[keepComp]) {
                keepComp = cid;
            }
        }
        if (localCompCount > 1) {
            std::vector<TriangleIdx> kept;
            kept.reserve(localCompSize[keepComp]);
            for (int triIdx = 0; triIdx < static_cast<int>(selectedTriangles.size()); ++triIdx) {
                if (localComp[triIdx] == keepComp) {
                    kept.push_back(selectedTriangles[triIdx]);
                } else {
                    droppedDisconnected++;
                }
            }
            selectedTriangles.swap(kept);
        }
    }

    // Step 6: Insert vertices and faces for OpenABF.
    std::unordered_map<int, std::size_t> gridToVertex;
    gridToVertex.reserve(selectedTriangles.size() * 2);

    std::size_t vertexIdx = 0;
    auto getOrInsertVertex = [&](int gridIdx) -> std::size_t {
        auto it = gridToVertex.find(gridIdx);
        if (it != gridToVertex.end()) {
            return it->second;
        }
        const int baseGridIdx = gridIdx < gridVertexCount ? gridIdx : syntheticToBaseGrid.at(gridIdx);
        const int row = baseGridIdx / points->cols;
        const int col = baseGridIdx % points->cols;
        const cv::Vec3f& pt = (*points)(row, col);

        OpenABF::Vec3d p;
        p[0] = pt[0];
        p[1] = pt[1];
        p[2] = pt[2];
        hem->insert_vertex(p);

        const std::size_t outIdx = vertexIdx++;
        gridToVertex.emplace(gridIdx, outIdx);
        vertexToGrid[outIdx] = baseGridIdx;
        return outIdx;
    };

    int faceCount = 0;
    try {
        for (const auto& t : selectedTriangles) {
            const std::size_t v0 = getOrInsertVertex(t.i0);
            const std::size_t v1 = getOrInsertVertex(t.i1);
            const std::size_t v2 = getOrInsertVertex(t.i2);
            std::vector<std::size_t> face = {v0, v1, v2};
            hem->insert_face(face);
            faceCount++;
        }
    } catch (const OpenABF::MeshException& e) {
        std::cerr << "ABF++: Failed to insert mesh faces: " << e.what() << std::endl;
        return cv::Mat_<cv::Vec2f>();
    }

    std::cout << "ABF++: Inserted " << vertexIdx << " vertices across 1 component(s)";
    if (componentCount > 1) {
        std::cout << " (selected largest of " << componentCount << ")";
    }
    std::cout << std::endl;
    std::cout << "ABF++: Inserted " << faceCount << " faces" << std::endl;
    if (skippedHardDegenerate > 0) {
        std::cout << "ABF++: Skipped " << skippedHardDegenerate
                  << " hard-degenerate triangles" << std::endl;
    }
    if (skippedLowQuality > 0) {
        std::cout << "ABF++: Skipped " << skippedLowQuality
                  << " low-quality triangles" << std::endl;
    }
    if (skippedDegenerateOut) {
        *skippedDegenerateOut = skippedHardDegenerate;
    }
    if (splitPinchVertices > 0) {
        std::cout << "ABF++: Split " << splitPinchVertices
                  << " boundary pinch vertex fan(s)" << std::endl;
    }
    if (droppedDisconnected > 0) {
        std::cout << "ABF++: Dropped " << droppedDisconnected
                  << " triangles from disconnected components" << std::endl;
    }
    if (faceCount == 0) {
        std::cerr << "ABF++: No non-degenerate faces found after filtering" << std::endl;
        return cv::Mat_<cv::Vec2f>();
    }

    try {
        hem->update_boundary();
    } catch (const OpenABF::MeshException& e) {
        std::cerr << "ABF++: Boundary update failed: " << e.what() << std::endl;
        return cv::Mat_<cv::Vec2f>();
    }

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
        if (uvs(row, col)[0] == -1.f) {
            uvs(row, col) = cv::Vec2f(
                static_cast<float>(v->pos[0]),
                static_cast<float>(v->pos[1])
            );
        }
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

static cv::Mat_<cv::Vec2f> runAbfWithRegridFallback(const QuadSurface& surface,
                                                    const ABFConfig& config)
{
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    if (!points || points->empty()) {
        return cv::Mat_<cv::Vec2f>();
    }

    auto countDegenerateTriangles = [](const cv::Mat_<cv::Vec3f>& pts) -> int {
        int degenerate = 0;
        for (int row = 0; row < pts.rows - 1; ++row) {
            for (int col = 0; col < pts.cols - 1; ++col) {
                const cv::Vec3f& p00 = pts(row, col);
                const cv::Vec3f& p01 = pts(row, col + 1);
                const cv::Vec3f& p10 = pts(row + 1, col);
                const cv::Vec3f& p11 = pts(row + 1, col + 1);
                if (!isValidSurfacePoint(p00) || !isValidSurfacePoint(p01) ||
                    !isValidSurfacePoint(p10) || !isValidSurfacePoint(p11)) {
                    continue;
                }
                if (!isUsableTriangle(p10, p00, p01)) {
                    degenerate++;
                }
                if (!isUsableTriangle(p10, p01, p11)) {
                    degenerate++;
                }
            }
        }
        return degenerate;
    };

    const int preDegenerate = countDegenerateTriangles(*points);
    if (preDegenerate > 0) {
        std::cout << "ABF++: Regridding surface before flatten (detected "
                  << preDegenerate << " degenerate triangles)" << std::endl;

        cv::Mat_<cv::Vec3f> regrid = regridPointsForFlattening(*points);
        cv::Mat_<cv::Vec3f>* regridPtr = new cv::Mat_<cv::Vec3f>(regrid);
        QuadSurface regridSurface(regridPtr, surface._scale);

        cv::Mat_<cv::Vec2f> pre = abfFlattenInternal(regridSurface, config, nullptr);
        if (!pre.empty()) {
            std::cout << "ABF++: Regrid-first flatten succeeded" << std::endl;
            return pre;
        }
        std::cerr << "ABF++: Regrid-first flatten failed, retrying with original mesh" << std::endl;
    }

    int skippedHardDegenerate = 0;
    cv::Mat_<cv::Vec2f> first = abfFlattenInternal(surface, config, &skippedHardDegenerate);
    if (!first.empty() && skippedHardDegenerate == 0) {
        return first;
    }

    std::cout << "ABF++: Regridding surface before retry (triggered by "
              << (first.empty() ? "flatten failure" : "hard-degenerate triangles")
              << ")" << std::endl;
    cv::Mat_<cv::Vec3f> regrid = regridPointsForFlattening(*points);
    cv::Mat_<cv::Vec3f>* regridPtr = new cv::Mat_<cv::Vec3f>(regrid);
    QuadSurface regridSurface(regridPtr, surface._scale);

    cv::Mat_<cv::Vec2f> retry = abfFlattenInternal(regridSurface, config, nullptr);
    if (!retry.empty()) {
        std::cout << "ABF++: Regrid retry succeeded" << std::endl;
        return retry;
    }

    if (!first.empty()) {
        std::cerr << "ABF++: Regrid retry failed; using first flatten result" << std::endl;
        return first;
    }

    std::cerr << "ABF++: Regrid retry failed" << std::endl;
    return cv::Mat_<cv::Vec2f>();
}

cv::Mat_<cv::Vec2f> abfFlatten(const QuadSurface& surface, const ABFConfig& config) {
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    if (!points || points->empty()) {
        std::cerr << "ABF++: Empty surface" << std::endl;
        return cv::Mat_<cv::Vec2f>();
    }

    // If no downsampling requested, run directly
    if (config.downsampleFactor <= 1) {
        cv::Mat_<cv::Vec2f> uvs = runAbfWithRegridFallback(surface, config);
        if (!uvs.empty() && config.alignToInputGrid) {
            alignUVsToInputGrid(*points, uvs);
        }
        return uvs;
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
    cv::Mat_<cv::Vec2f> coarseUVs = runAbfWithRegridFallback(coarseSurface, config);

    if (coarseUVs.empty()) {
        return cv::Mat_<cv::Vec2f>();
    }

    // Upsample UVs back to original resolution
    std::cout << "ABF++: Upsampling UVs from " << coarseUVs.rows << "x" << coarseUVs.cols
              << " to " << originalRows << "x" << originalCols << std::endl;

    cv::Mat_<cv::Vec2f> result = upsampleUVs(coarseUVs, originalRows, originalCols, config.downsampleFactor);
    if (!result.empty() && config.alignToInputGrid) {
        alignUVsToInputGrid(*points, result);
    }
    return result;
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

static float percentileFromSorted(const std::vector<float>& sorted, float q)
{
    if (sorted.empty()) {
        return 0.f;
    }
    q = std::clamp(q, 0.f, 1.f);
    const std::size_t idx = static_cast<std::size_t>(q * static_cast<float>(sorted.size() - 1));
    return sorted[idx];
}

static float triangleStretchMax(const cv::Vec3f& p0, const cv::Vec3f& p1, const cv::Vec3f& p2,
                                const cv::Vec2f& uv0, const cv::Vec2f& uv1, const cv::Vec2f& uv2)
{
    constexpr float kMinEdgeSq = 1e-12f;
    auto edgeStretch = [&](const cv::Vec3f& a3, const cv::Vec3f& b3,
                           const cv::Vec2f& a2, const cv::Vec2f& b2) -> float {
        const cv::Vec3f d3 = b3 - a3;
        const cv::Vec2f d2 = b2 - a2;
        const float len3Sq = d3.dot(d3);
        const float len2Sq = d2.dot(d2);
        if (!std::isfinite(len3Sq) || !std::isfinite(len2Sq) || len3Sq <= kMinEdgeSq || len2Sq < 0.f) {
            return std::numeric_limits<float>::infinity();
        }
        return std::sqrt(len2Sq / len3Sq);
    };

    const float s01 = edgeStretch(p0, p1, uv0, uv1);
    const float s12 = edgeStretch(p1, p2, uv1, uv2);
    const float s20 = edgeStretch(p2, p0, uv2, uv0);
    return std::max({s01, s12, s20});
}

QuadSurface* abfFlattenToNewSurface(const QuadSurface& surface, const ABFConfig& config) {
    // Step 1: Compute flattened UVs
    cv::Mat_<cv::Vec2f> uvs = abfFlatten(surface, config);
    if (uvs.empty()) {
        return nullptr;
    }

    const cv::Mat_<cv::Vec3f>* srcPoints = surface.rawPointsPtr();
    if (!srcPoints || srcPoints->empty()) {
        return nullptr;
    }

    // Track valid source bounds. Rasterized points are barycentric mixtures of source
    // vertices, so they must remain within these limits (up to tiny numeric tolerance).
    cv::Vec3f srcMin(std::numeric_limits<float>::max(),
                     std::numeric_limits<float>::max(),
                     std::numeric_limits<float>::max());
    cv::Vec3f srcMax(std::numeric_limits<float>::lowest(),
                     std::numeric_limits<float>::lowest(),
                     std::numeric_limits<float>::lowest());
    bool haveSrcBounds = false;
    for (int row = 0; row < srcPoints->rows; ++row) {
        for (int col = 0; col < srcPoints->cols; ++col) {
            const cv::Vec3f& p = (*srcPoints)(row, col);
            if (p[0] == -1.f || !std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) {
                continue;
            }
            haveSrcBounds = true;
            srcMin[0] = std::min(srcMin[0], p[0]);
            srcMin[1] = std::min(srcMin[1], p[1]);
            srcMin[2] = std::min(srcMin[2], p[2]);
            srcMax[0] = std::max(srcMax[0], p[0]);
            srcMax[1] = std::max(srcMax[1], p[1]);
            srcMax[2] = std::max(srcMax[2], p[2]);
        }
    }
    if (!haveSrcBounds) {
        return nullptr;
    }

    // Step 2: Estimate stretch statistics and reject extreme UV outlier triangles.
    auto hasValidUV = [](const cv::Vec2f& uv) -> bool {
        return uv[0] != -1.f && std::isfinite(uv[0]) && std::isfinite(uv[1]);
    };
    std::vector<float> stretchSamples;
    stretchSamples.reserve(static_cast<std::size_t>((srcPoints->rows - 1) * (srcPoints->cols - 1) * 2));

    for (int row = 0; row < srcPoints->rows - 1; ++row) {
        for (int col = 0; col < srcPoints->cols - 1; ++col) {
            const cv::Vec3f& p00 = (*srcPoints)(row, col);
            const cv::Vec3f& p01 = (*srcPoints)(row, col + 1);
            const cv::Vec3f& p10 = (*srcPoints)(row + 1, col);
            const cv::Vec3f& p11 = (*srcPoints)(row + 1, col + 1);
            if (!isValidSurfacePoint(p00) || !isValidSurfacePoint(p01) ||
                !isValidSurfacePoint(p10) || !isValidSurfacePoint(p11)) {
                continue;
            }

            const cv::Vec2f& uv00 = uvs(row, col);
            const cv::Vec2f& uv01 = uvs(row, col + 1);
            const cv::Vec2f& uv10 = uvs(row + 1, col);
            const cv::Vec2f& uv11 = uvs(row + 1, col + 1);
            if (!hasValidUV(uv00) || !hasValidUV(uv01) || !hasValidUV(uv10) || !hasValidUV(uv11)) {
                continue;
            }

            const float s1 = triangleStretchMax(p10, p00, p01, uv10, uv00, uv01);
            const float s2 = triangleStretchMax(p10, p01, p11, uv10, uv01, uv11);
            if (std::isfinite(s1)) {
                stretchSamples.push_back(s1);
            }
            if (std::isfinite(s2)) {
                stretchSamples.push_back(s2);
            }
        }
    }

    float maxAllowedStretch = std::numeric_limits<float>::infinity();
    bool useStretchFilter = false;
    if (stretchSamples.size() >= 128) {
        std::sort(stretchSamples.begin(), stretchSamples.end());
        const float p50 = percentileFromSorted(stretchSamples, 0.50f);
        const float p95 = percentileFromSorted(stretchSamples, 0.95f);
        const float p99 = percentileFromSorted(stretchSamples, 0.99f);
        const float tailRatio = p99 / std::max(p50, 1e-12f);
        if (std::isfinite(tailRatio) && tailRatio > 6.f) {
            // Keep normal distortion, reject only very extreme tails that create spikes.
            maxAllowedStretch = std::max({p99 * 8.f, p95 * 12.f, p50 * 32.f});
            useStretchFilter = std::isfinite(maxAllowedStretch) && maxAllowedStretch > 0.f;
            if (useStretchFilter) {
                std::cout << "ABF++: Enabling UV stretch outlier filter (p50=" << p50
                          << ", p99=" << p99
                          << ", max=" << maxAllowedStretch << ")" << std::endl;
            }
        }
    }

    cv::Mat_<uint8_t> uvUsed(srcPoints->rows, srcPoints->cols, uint8_t(0));
    int rejectedTriangles = 0;
    for (int row = 0; row < srcPoints->rows - 1; ++row) {
        for (int col = 0; col < srcPoints->cols - 1; ++col) {
            const cv::Vec3f& p00 = (*srcPoints)(row, col);
            const cv::Vec3f& p01 = (*srcPoints)(row, col + 1);
            const cv::Vec3f& p10 = (*srcPoints)(row + 1, col);
            const cv::Vec3f& p11 = (*srcPoints)(row + 1, col + 1);
            if (!isValidSurfacePoint(p00) || !isValidSurfacePoint(p01) ||
                !isValidSurfacePoint(p10) || !isValidSurfacePoint(p11)) {
                continue;
            }

            const cv::Vec2f& uv00 = uvs(row, col);
            const cv::Vec2f& uv01 = uvs(row, col + 1);
            const cv::Vec2f& uv10 = uvs(row + 1, col);
            const cv::Vec2f& uv11 = uvs(row + 1, col + 1);
            if (!hasValidUV(uv00) || !hasValidUV(uv01) || !hasValidUV(uv10) || !hasValidUV(uv11)) {
                continue;
            }

            bool acceptTri1 = true;
            bool acceptTri2 = true;
            if (useStretchFilter) {
                const float s1 = triangleStretchMax(p10, p00, p01, uv10, uv00, uv01);
                const float s2 = triangleStretchMax(p10, p01, p11, uv10, uv01, uv11);
                acceptTri1 = std::isfinite(s1) && s1 <= maxAllowedStretch;
                acceptTri2 = std::isfinite(s2) && s2 <= maxAllowedStretch;
                rejectedTriangles += (!acceptTri1 ? 1 : 0) + (!acceptTri2 ? 1 : 0);
            }

            if (acceptTri1) {
                uvUsed(row + 1, col) = 1;     // p10
                uvUsed(row, col) = 1;         // p00
                uvUsed(row, col + 1) = 1;     // p01
            }
            if (acceptTri2) {
                uvUsed(row + 1, col) = 1;     // p10
                uvUsed(row, col + 1) = 1;     // p01
                uvUsed(row + 1, col + 1) = 1; // p11
            }
        }
    }

    // Step 3: Find UV bounds from triangles that survived outlier filtering.
    float uvMinX = std::numeric_limits<float>::max();
    float uvMinY = std::numeric_limits<float>::max();
    float uvMaxX = std::numeric_limits<float>::lowest();
    float uvMaxY = std::numeric_limits<float>::lowest();
    bool haveUvBounds = false;
    for (int row = 0; row < uvs.rows; ++row) {
        for (int col = 0; col < uvs.cols; ++col) {
            if (uvUsed(row, col) == 0) {
                continue;
            }
            const cv::Vec2f& uv = uvs(row, col);
            if (!hasValidUV(uv)) {
                continue;
            }
            haveUvBounds = true;
            uvMinX = std::min(uvMinX, uv[0]);
            uvMinY = std::min(uvMinY, uv[1]);
            uvMaxX = std::max(uvMaxX, uv[0]);
            uvMaxY = std::max(uvMaxY, uv[1]);
        }
    }

    if (!haveUvBounds) {
        // Fallback: if filtering removed everything, keep legacy behavior.
        for (int row = 0; row < uvs.rows; ++row) {
            for (int col = 0; col < uvs.cols; ++col) {
                const cv::Vec2f& uv = uvs(row, col);
                if (!hasValidUV(uv)) {
                    continue;
                }
                uvMinX = std::min(uvMinX, uv[0]);
                uvMinY = std::min(uvMinY, uv[1]);
                uvMaxX = std::max(uvMaxX, uv[0]);
                uvMaxY = std::max(uvMaxY, uv[1]);
                haveUvBounds = true;
            }
        }
        useStretchFilter = false;
        rejectedTriangles = 0;
    }
    if (!haveUvBounds) {
        return nullptr;
    }

    if (rejectedTriangles > 0) {
        std::cout << "ABF++: Rejected " << rejectedTriangles
                  << " UV outlier triangle(s) before rasterization" << std::endl;
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

    // Step 5: Rasterize triangles onto the output grid.
    // Precompute UV-to-grid transform factors
    const float rxInv = (gridW - 1) / std::max(uvRange[0], 1e-12f);
    const float ryInv = (gridH - 1) / std::max(uvRange[1], 1e-12f);

    const float srcBoundsEps = 1e-3f;
    const auto isPlausiblePos = [&](const cv::Vec3f& pos) -> bool {
        if (!std::isfinite(pos[0]) || !std::isfinite(pos[1]) || !std::isfinite(pos[2])) {
            return false;
        }
        return pos[0] >= srcMin[0] - srcBoundsEps && pos[0] <= srcMax[0] + srcBoundsEps &&
               pos[1] >= srcMin[1] - srcBoundsEps && pos[1] <= srcMax[1] + srcBoundsEps &&
               pos[2] >= srcMin[2] - srcBoundsEps && pos[2] <= srcMax[2] + srcBoundsEps;
    };

    // For each valid quad in the source, triangulate and rasterize.
    // This loop intentionally stays single-threaded because triangles overlap at
    // shared edges, and unsynchronized parallel writes can corrupt output pixels.
    for (int row = 0; row < srcPoints->rows - 1; ++row) {
        for (int col = 0; col < srcPoints->cols - 1; ++col) {
            const cv::Vec3f& p00 = (*srcPoints)(row, col);
            const cv::Vec3f& p01 = (*srcPoints)(row, col + 1);
            const cv::Vec3f& p10 = (*srcPoints)(row + 1, col);
            const cv::Vec3f& p11 = (*srcPoints)(row + 1, col + 1);

            if (!isValidSurfacePoint(p00) || !isValidSurfacePoint(p01) ||
                !isValidSurfacePoint(p10) || !isValidSurfacePoint(p11))
                continue;

            const cv::Vec2f& uv00 = uvs(row, col);
            const cv::Vec2f& uv01 = uvs(row, col + 1);
            const cv::Vec2f& uv10 = uvs(row + 1, col);
            const cv::Vec2f& uv11 = uvs(row + 1, col + 1);
            if (!hasValidUV(uv00) || !hasValidUV(uv01) || !hasValidUV(uv10) || !hasValidUV(uv11)) {
                continue;
            }

            // Transform UVs to grid coordinates (inlined for performance)
            cv::Vec2f guv00((uv00[0] - uvMin[0]) * rxInv, (uv00[1] - uvMin[1]) * ryInv);
            cv::Vec2f guv01((uv01[0] - uvMin[0]) * rxInv, (uv01[1] - uvMin[1]) * ryInv);
            cv::Vec2f guv10((uv10[0] - uvMin[0]) * rxInv, (uv10[1] - uvMin[1]) * ryInv);
            cv::Vec2f guv11((uv11[0] - uvMin[0]) * rxInv, (uv11[1] - uvMin[1]) * ryInv);

            bool acceptTri1 = true;
            bool acceptTri2 = true;
            if (useStretchFilter) {
                const float s1 = triangleStretchMax(p10, p00, p01, uv10, uv00, uv01);
                const float s2 = triangleStretchMax(p10, p01, p11, uv10, uv01, uv11);
                acceptTri1 = std::isfinite(s1) && s1 <= maxAllowedStretch;
                acceptTri2 = std::isfinite(s2) && s2 <= maxAllowedStretch;
            }

            // Rasterize Triangle 1: (p10, p00, p01) with UVs (guv10, guv00, guv01)
            if (acceptTri1) {
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
                                if (isPlausiblePos(pos) && (*outPoints)(y, x)[0] == -1.f) {
                                    (*outPoints)(y, x) = pos;
                                }
                            }
                        }
                    }
                }
            }

            // Rasterize Triangle 2: (p10, p01, p11) with UVs (guv10, guv01, guv11)
            if (acceptTri2) {
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
                                if (isPlausiblePos(pos) && (*outPoints)(y, x)[0] == -1.f) {
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
