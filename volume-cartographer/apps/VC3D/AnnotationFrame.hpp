#pragma once

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
    // the volume's".
    std::optional<double> voxelSizeUm;
    // What carries the volume's own voxel counts into that frame, i.e. the
    // ratio of the two resolutions. 1.0 when the grids coincide.
    double factor = 1.0;
    // The volume's dimensions in that frame; zeroes when the dimensions handed
    // in were unusable.
    std::array<double, 3> extentXyz{0.0, 0.0, 0.0};
};

// Resolves that frame from what a volume can report about itself:
// `volumeVoxelSizeUm` and `volumeDimsXyz` are the store's own, and
// `stampedSourceResolutionUm` is an open-data coordinate identity's
// sourceOriginalResolution where one exists.
//
// The factor is deliberately the ratio of the two resolutions rather than a
// power of two read off a pyramid level. A stamped source resolution need not be
// a downsample of this particular store, so composing levels is ambiguous
// exactly when both mechanisms are in play, whereas the ratio is correct by
// construction: it is the number that makes the resolution and the voxel counts
// describe one grid. Pairing counts from one grid with a resolution from another
// is the mismatch this exists to prevent.
inline AnnotationFrame deriveAnnotationFrame(
    double volumeVoxelSizeUm,
    int baseScaleLevel,
    std::optional<double> stampedSourceResolutionUm,
    const std::array<double, 3>& volumeDimsXyz)
{
    AnnotationFrame frame;
    const bool haveVolumeVoxel =
        std::isfinite(volumeVoxelSizeUm) && volumeVoxelSizeUm > 0.0;

    if (stampedSourceResolutionUm &&
        std::isfinite(*stampedSourceResolutionUm) &&
        *stampedSourceResolutionUm > 0.0) {
        // A stamped resolution is an absolute statement about the annotated
        // frame, so it outranks anything inferred from where this store sits in
        // its own pyramid — including a store reporting baseScaleLevel() == 0
        // whose coordinates are nonetheless a finer grid than its own voxel
        // size describes.
        frame.voxelSizeUm = *stampedSourceResolutionUm;
    } else if (haveVolumeVoxel && baseScaleLevel > 0) {
        // Untagged and rebased: voxelSize() already carries the rebase, so
        // dividing it back out recovers the store's level-0 resolution, which
        // with nothing else to go on is the annotated frame.
        frame.voxelSizeUm =
            volumeVoxelSizeUm / std::pow(2.0, static_cast<double>(baseScaleLevel));
    } else if (haveVolumeVoxel) {
        frame.voxelSizeUm = volumeVoxelSizeUm;
    }

    if (haveVolumeVoxel && frame.voxelSizeUm && *frame.voxelSizeUm > 0.0) {
        const double ratio = volumeVoxelSizeUm / *frame.voxelSizeUm;
        if (std::isfinite(ratio) && ratio > 0.0)
            frame.factor = ratio;
    }

    const bool haveDims =
        std::isfinite(volumeDimsXyz[0]) && volumeDimsXyz[0] > 0.0 &&
        std::isfinite(volumeDimsXyz[1]) && volumeDimsXyz[1] > 0.0 &&
        std::isfinite(volumeDimsXyz[2]) && volumeDimsXyz[2] > 0.0;
    if (haveDims) {
        frame.extentXyz = {volumeDimsXyz[0] * frame.factor,
                           volumeDimsXyz[1] * frame.factor,
                           volumeDimsXyz[2] * frame.factor};
    }
    return frame;
}

} // namespace vc3d::annotation
