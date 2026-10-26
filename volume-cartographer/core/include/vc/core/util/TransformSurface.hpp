#pragma once
// Whole-surface coordinate transforms for the tifxyz file tools: an affine
// between two uniform scales, regridding to the transformed spacing, and a
// uniform scale that keeps the grid.

#include <array>
#include <filesystem>
#include <optional>

#include <vc_reader/affine.hpp>

class QuadSurface;

namespace vc::core::util {

// Scale, apply `matrix`, scale again; points that leave float range become
// invalid. The grid is then resampled so its spacing follows the transform.
void transformSurfacePoints(QuadSurface* surface,
                            double scaleBeforeAffine,
                            const std::optional<vcr::catalog::Affine>& matrix,
                            double scaleAfterAffine);
// Recompute the cached bbox and the meta bbox and scale after the points changed.
void refreshTransformedSurfaceState(QuadSurface* surface);
// Uniform coordinate-unit conversion, preserving grid samples and ancillary channels.
// Supply the actual destination shape when available (pyramid rounding is not invertible).
void scaleSurfaceWithoutResampling(QuadSurface* surface, double factor,
    std::optional<std::array<int, 3>> targetShapeZYX = std::nullopt);
// Streams float32 XYZ TIFFs (tiled or stripped) to an uncompressed stripped output.
// Never overwrites an existing destination. Publishes the directory only on success.
void scaleTifxyzWithoutResampling(const std::filesystem::path& input,
    const std::filesystem::path& output, double factor,
    std::optional<std::array<int, 3>> targetShapeZYX = std::nullopt);

} // namespace vc::core::util
