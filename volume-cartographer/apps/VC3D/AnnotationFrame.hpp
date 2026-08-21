#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include <optional>

namespace vc3d::annotation {

// The grid a volume's annotations index, which is not in general the grid the
// volume itself is stored on: annotations are made in a source frame that a
// downsampled or rebased store describes only indirectly.
struct AnnotationFrame {
    // Resolution the annotated coordinates are expressed at. Absent when
    // nothing available can say, which callers must not read as "the same as
    // the volume's". Physical metadata only: it never participates in deriving
    // the coordinate factor below.
    std::optional<double> voxelSizeUm;
    // What carries the volume's own voxel counts into that frame. 1.0 when the
    // grids coincide. Meaningful only where extentXyz is non-zero.
    double factor = 1.0;
    // The volume's dimensions in that frame; zeroes when the dimensions handed
    // in were unusable, and also when the factor is unknown — raw counts are
    // not counts in this frame.
    std::array<double, 3> extentXyz{0.0, 0.0, 0.0};
};

// Resolves that frame from what a volume can report about itself:
// `volumeVoxelSizeUm` and `volumeDimsXyz` are the store's own,
// `exactCoordinateFactor` is an open-data coordinate identity's
// sourceCoordinateScaleFactor where one exists, and
// `stampedSourceResolutionUm` is that identity's sourceOriginalResolution.
//
// Coordinate extents and physical resolution are deliberately derived
// separately, and the exact factor outranks any resolution ratio. A stamped
// resolution is a physical figure that may be rounded (9.596 µm against a
// 2.4 µm source gives 3.99833), while the coordinate factor is an exact
// integer statement about the grids; deriving the extents from the rounded
// ratio hands consumers a fractional grid that an exact integer-rescale test
// then rightly refuses — a conflict the derivation itself would have invented.
// The resolution stays what it is: metadata for display and for voxel-size
// scale derivation, never a way to compute counts.
//
// Without an exact factor the ratio of the two resolutions is still honoured
// (a stamped resolution need not be a downsample of this store, so pyramid
// arithmetic is ambiguous exactly when both mechanisms are in play), and an
// untagged rebased store lifts its counts by its own pyramid position.
//
// A store that states a foreign resolution but cannot say its own has no
// derivable factor at all: its counts stay zero rather than being presented as
// counts in a frame nothing could carry them into.
inline AnnotationFrame deriveAnnotationFrame(
    double volumeVoxelSizeUm,
    int baseScaleLevel,
    std::optional<double> exactCoordinateFactor,
    std::optional<double> stampedSourceResolutionUm,
    const std::array<double, 3>& volumeDimsXyz)
{
    AnnotationFrame frame;
    const bool haveVolumeVoxel =
        std::isfinite(volumeVoxelSizeUm) && volumeVoxelSizeUm > 0.0;
    const bool haveStampedResolution = stampedSourceResolutionUm &&
                                       std::isfinite(*stampedSourceResolutionUm) &&
                                       *stampedSourceResolutionUm > 0.0;
    const bool haveExactFactor = exactCoordinateFactor &&
                                 std::isfinite(*exactCoordinateFactor) &&
                                 *exactCoordinateFactor > 0.0;

    // The factor, in descending order of trust. Exact statements first; a
    // ratio of physical resolutions only where nothing exact was said.
    std::optional<double> factor;
    if (haveExactFactor) {
        factor = *exactCoordinateFactor;
    } else if (haveStampedResolution) {
        // A stamp outranks the store's own pyramid position, so a stamped
        // store never falls through to level arithmetic — including one whose
        // own voxel size is unusable, where the stamp claims a frame nothing
        // can relate this store's counts to and the factor stays unknown.
        if (haveVolumeVoxel) {
            const double ratio = volumeVoxelSizeUm / *stampedSourceResolutionUm;
            if (std::isfinite(ratio) && ratio > 0.0) {
                factor = ratio;
            }
        }
    } else if (baseScaleLevel > 0) {
        // Untagged and rebased: the store's own pyramid position is an exact
        // integer statement about its level-0 grid.
        factor = std::pow(2.0, static_cast<double>(baseScaleLevel));
    } else {
        // Nothing claims any frame but the store's own.
        factor = 1.0;
    }

    if (haveStampedResolution) {
        // A stamped resolution is an absolute statement about the annotated
        // frame, so it outranks anything inferred from where this store sits
        // in its own pyramid.
        frame.voxelSizeUm = *stampedSourceResolutionUm;
    } else if (haveVolumeVoxel && factor) {
        frame.voxelSizeUm = volumeVoxelSizeUm / *factor;
    }

    const bool haveDims =
        std::isfinite(volumeDimsXyz[0]) && volumeDimsXyz[0] > 0.0 &&
        std::isfinite(volumeDimsXyz[1]) && volumeDimsXyz[1] > 0.0 &&
        std::isfinite(volumeDimsXyz[2]) && volumeDimsXyz[2] > 0.0;
    if (haveDims && factor) {
        frame.factor = *factor;
        frame.extentXyz = {volumeDimsXyz[0] * *factor,
                           volumeDimsXyz[1] * *factor,
                           volumeDimsXyz[2] * *factor};
    }
    return frame;
}

// True when two frames describe the same grid, i.e. when geometry built in one is
// still valid in the other. Two stores of one scan at different pyramid levels
// compare equal, which is what lets switching between them leave derived geometry
// alone; a store of a different scan does not.
//
// Voxel sizes compare with a relative tolerance, so metadata that round-trips 2.4
// slightly differently is not mistaken for a new scan; extents compare as the
// integer voxel counts they are.
inline bool sameAnnotationFrame(const AnnotationFrame& a,
                                const AnnotationFrame& b)
{
    if (a.voxelSizeUm.has_value() != b.voxelSizeUm.has_value())
        return false;
    if (a.voxelSizeUm) {
        const double scale = std::max(*a.voxelSizeUm, *b.voxelSizeUm);
        if (std::abs(*a.voxelSizeUm - *b.voxelSizeUm) > 1e-6 * scale)
            return false;
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (std::llround(a.extentXyz[axis]) != std::llround(b.extentXyz[axis]))
            return false;
    }
    return true;
}

// Whether two frames are the same *grid*, i.e. whether geometry expressed in
// voxels indexes the same voxels in both. Voxel counts only.
//
// Neither this nor sameAnnotationFrame proves the two are voxelwise aligned:
// AnnotationFrame carries no origin, axis order or pose, so equal counts are also
// consistent with a crop or a registration. Both are proxies. This is the
// narrower question, asked where the answer decides whether to destroy work: a
// differing physical voxel size relabels a grid, it does not replace it, and
// geometry computed in voxels is still about the same voxels.
inline bool sameAnnotationGrid(const AnnotationFrame& a, const AnnotationFrame& b)
{
    for (int axis = 0; axis < 3; ++axis) {
        if (std::llround(a.extentXyz[axis]) != std::llround(b.extentXyz[axis]))
            return false;
    }
    return true;
}

} // namespace vc3d::annotation
