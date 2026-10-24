#pragma once

#include "vc/lasagna/Manifest.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace vc3d::line_annotation {

// Resolve legacy metadata once. Later volume/dataset selections must not
// change the coordinate grid of unchanged stored geometry.
template <typename Resolve>
const std::optional<std::array<std::size_t, 3>>& retainFiberCoordinateShape(
    std::optional<std::array<std::size_t, 3>>& shape, Resolve&& resolve)
{
    if (!shape) {
        const auto resolved = resolve();
        if (!resolved || std::any_of(resolved->begin(), resolved->end(),
                                    [](std::size_t n) { return n == 0; }))
            throw std::runtime_error("Cannot resolve a positive fiber coordinate shape");
        shape = resolved;
    }
    return shape;
}

// A viewer-created fiber uses the viewer's volume grid, not an inferred
// level-0 grid (whose dimensions cannot be recovered exactly by scaling).
inline std::array<std::size_t, 3> fiberCoordinateShapeFromVolume(
    const std::array<int, 3>& shapeZYX)
{
    std::array<std::size_t, 3> result{};
    for (std::size_t axis = 0; axis < result.size(); ++axis) {
        if (shapeZYX[axis] <= 0)
            throw std::runtime_error("Fiber creation requires a positive volume shape");
        result[axis] = static_cast<std::size_t>(shapeZYX[axis]);
    }
    return result;
}

struct FiberNormalCoordinateScales {
    double fiberBaseToNormalBase = 1.0;
    double traceToNormalBase = 1.0;
};

// Fiber JSON geometry is canonical in the fiber manifest's base grid, while
// an annotation viewer may be rendering a downsampled volume. Return the
// factor that carries stored fiber coordinates into that volume's grid.
inline double resolveFiberBaseToVolumeScale(
    const std::optional<std::array<std::size_t, 3>>& fiberBaseShapeZYX,
    const std::array<int, 3>& volumeShapeZYX)
{
    if (!fiberBaseShapeZYX) {
        return 1.0;
    }
    std::array<std::size_t, 3> working{};
    for (std::size_t axis = 0; axis < working.size(); ++axis) {
        if (volumeShapeZYX[axis] <= 0) {
            throw std::runtime_error("active volume shape must be positive");
        }
        working[axis] = static_cast<std::size_t>(volumeShapeZYX[axis]);
    }
    return vc::lasagna::dyadicCoordinateScaleBetweenShapes(
        *fiberBaseShapeZYX, working, 5);
}

// The line geometry is stored in the fiber manifest's base coordinates, while
// its normal field may have been published against another level of the same
// volume pyramid. Return the runtime adapters expected by LasagnaDataset.
inline FiberNormalCoordinateScales resolveFiberNormalCoordinateScales(
    const std::optional<std::array<std::size_t, 3>>& normalBaseShapeZYX,
    const std::optional<std::array<std::size_t, 3>>& fiberBaseShapeZYX,
    double traceToFiberBaseScale)
{
    if (!(traceToFiberBaseScale > 0.0) ||
        !std::isfinite(traceToFiberBaseScale)) {
        throw std::runtime_error(
            "fiber trace-to-base scale must be positive and finite");
    }

    double fiberBaseToNormalBase = 1.0;
    if (normalBaseShapeZYX && fiberBaseShapeZYX) {
        fiberBaseToNormalBase = vc::lasagna::dyadicCoordinateScaleBetweenShapes(
            *fiberBaseShapeZYX, *normalBaseShapeZYX, 5);
    }
    const double traceToNormalBase =
        traceToFiberBaseScale * fiberBaseToNormalBase;
    if (!(traceToNormalBase > 0.0) || !std::isfinite(traceToNormalBase)) {
        throw std::runtime_error(
            "composed trace-to-normal scale must be positive and finite");
    }
    return {
        fiberBaseToNormalBase,
        traceToNormalBase,
    };
}

// Manifest locations to consult, in order, for a fiber that does not store
// its own coordinate base shape: the manifest its trace spans recorded, then
// the package's selected fiber-inference dataset. Older fibers recorded
// absolute manifest paths that may since have moved, so the package selection
// (which is what traces them today) is the fallback.
inline std::vector<std::string> fiberBaseShapeManifestCandidates(
    const std::string& fiberManifestLocation,
    const std::string& selectedFiberInferenceDataset)
{
    std::vector<std::string> candidates;
    if (!fiberManifestLocation.empty()) {
        candidates.push_back(fiberManifestLocation);
    }
    if (!selectedFiberInferenceDataset.empty() &&
        selectedFiberInferenceDataset != fiberManifestLocation) {
        candidates.push_back(selectedFiberInferenceDataset);
    }
    return candidates;
}

} // namespace vc3d::line_annotation
