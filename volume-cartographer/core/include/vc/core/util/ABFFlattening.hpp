#pragma once

#include <opencv2/core.hpp>
#include <cstddef>

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

    /** If true, rotate output so highest Z values are at row 0 (default: true) */
    bool rotateHighZToTop = true;
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
