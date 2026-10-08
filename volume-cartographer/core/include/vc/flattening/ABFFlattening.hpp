#pragma once

#include <opencv2/core/mat.hpp>
#include <cstddef>
#include <string>
#include <vector>

// Forward declaration
class QuadSurface;

namespace vc {

/**
 * @brief Configuration for ABF++ mesh flattening
 */
struct ABFConfig {
    /** Maximum iterations for ABF++ angle optimization (default: 10) */
    std::size_t maxIterations = 10;

    /** If true, run ABF++ optimization before LSCM. If false, use only LSCM */
    bool useABF = true;

    /** If true, scale the output to match the original 3D surface area */
    bool scaleToOriginalArea = true;

    /**
     * Downsample factor for ABF++ computation (default: 1 = full resolution).
     * Higher values (2, 4, 8) reduce grid resolution before ABF++ for faster
     * computation, then interpolate UVs back to full resolution.
     * Factor 2 = half resolution (4x faster), Factor 4 = quarter (16x faster).
     */
    int downsampleFactor = 1;

    /**
     * If true, normalize UV orientation to follow input grid directions
     * (increasing column -> +U, increasing row -> +V). This removes large
     * arbitrary global rotations/flips from ABF/LSCM.
     */
    bool alignToInputGrid = false;

    /** If true, rotate output so highest Z values are at row 0 (default: false) */
    bool rotateHighZToTop = false;
};

struct ABFVertexIssue {
    cv::Vec2i point = {-1, -1};
    float score = 0.f;
};

struct ABFScaleConsistency {
    bool checked = false;
    bool consistent = true;
    bool checkedColumns = false;
    bool checkedRows = false;
    float columnMedianSpacingVoxels = 0.f;
    float rowMedianSpacingVoxels = 0.f;
    float columnScaleRatio = 1.f;
    float rowScaleRatio = 1.f;
    std::string failureReason;
};

/**
 * @brief Check that tifxyz scale agrees with measured grid spacing.
 *
 * TIFXYZ scale is stored as grid cells per voxel, so a representative
 * one-cell spacing in voxels multiplied by the corresponding scale should be
 * approximately 1.  The check is deliberately generous and sampling-bounded:
 * it is a last-line guard against malformed metadata causing pathological
 * output allocations, not a replacement for detailed surface preflight.
 *
 * Axes with too few valid adjacent samples are skipped.  If neither axis has
 * enough samples, checked is false and consistent remains true.
 */
ABFScaleConsistency checkAbfInputScale(const cv::Mat_<cv::Vec3f>& points,
                                       const cv::Vec2f& scale,
                                       float toleranceFactor = 4.f);

struct ABFDiagnostics {
    bool success = false;
    bool exploded = false;
    std::string failureReason;
    std::size_t abfIterations = 0;
    double abfGradient = 0.0;
    int validUvCount = 0;
    int flippedTriangles = 0;
    int nearZeroUvTriangles = 0;
    int crowdedUvPairs = 0;
    float maxVertexBadness = 0.f;
    cv::Mat_<cv::Vec2f> uv;
    cv::Mat_<float> vertexBadness;
    std::vector<ABFVertexIssue> worstVertices;
};

/**
 * @brief Flatten a QuadSurface mesh using ABF++ (Angle-Based Flattening)
 *
 * Computes a low-distortion 2D parameterization of the surface.
 * The mesh is triangulated, then ABF++ optimizes vertex angles to minimize
 * angular distortion, followed by LSCM to compute final UV coordinates.
 *
 * @param surface Input surface to flatten
 * @param config Flattening configuration
 * @return cv::Mat_<cv::Vec2f> UV coordinates matching grid layout, or empty on failure
 */
cv::Mat_<cv::Vec2f> abfFlatten(const QuadSurface& surface, const ABFConfig& config = {});

/**
 * @brief Run ABF/LSCM and return per-grid-vertex diagnostics.
 *
 * The returned badness combines ABF angle-constraint diagnostics with
 * post-LSCM UV sanity checks such as flipped/near-zero triangles and extreme
 * stretch/compression. Grid coordinates are row/col in the provided surface.
 */
ABFDiagnostics diagnoseAbfFlattening(const QuadSurface& surface,
                                     const ABFConfig& config = {},
                                     std::size_t maxWorstVertices = 64);

/**
 * @brief Flatten and store UVs in the surface's "uv" channel
 *
 * @param surface Surface to flatten (modified in place)
 * @param config Flattening configuration
 * @return true if flattening succeeded
 */
bool abfFlattenInPlace(QuadSurface& surface, const ABFConfig& config = {});

/**
 * @brief Create a new surface with positions rearranged according to flattened UVs
 *
 * This is useful for rendering: the grid layout of the new surface matches
 * the computed UV parameterization, so rendering produces a distortion-corrected
 * texture. This is the same transformation that vc_obj2tifxyz performs when
 * loading a flattened OBJ mesh.
 *
 * @param surface Input surface to flatten
 * @param config Flattening configuration
 * @return QuadSurface* New surface with rearranged positions, or nullptr on failure.
 *         Caller takes ownership.
 */
QuadSurface* abfFlattenToNewSurface(const QuadSurface& surface, const ABFConfig& config = {});

} // namespace vc
