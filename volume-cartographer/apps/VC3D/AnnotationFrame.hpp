#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "vc/core/util/ScrollUmbilicus.hpp"
#include <optional>
#include <string>

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
    // The volume's dimensions in that frame; zeroes when the dimensions handed in
    // were unusable, and also when the factor is unknown — raw counts are not
    // counts in this frame.
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
    // The factor is only known when the store could say what its own voxels are.
    // Without it the counts cannot be carried into the annotated frame at all, and
    // presenting the raw counts as though they were already in that frame is the
    // fail-open this whole derivation exists to close.
    if (haveDims && haveVolumeVoxel && frame.voxelSizeUm) {
        frame.extentXyz = {volumeDimsXyz[0] * frame.factor,
                           volumeDimsXyz[1] * frame.factor,
                           volumeDimsXyz[2] * frame.factor};
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

// How a set of generated views got its scroll-centre reference.
enum class UmbilicusOrientationMode {
    // No usable umbilicus: the volume's raw XY centre.
    VolumeCentre,
    // A resolved umbilicus carried into the annotation frame by a derived scale.
    Applied,
    // A resolved umbilicus read the pre-metadata way, optionally through a
    // registration transform's inverse.
    Legacy,
};

// Everything a materialized set of generated views took its orientation from.
// Recorded when the views are built and compared when something changes, so the
// question "do these views still reflect the current state" is asked of the
// actual inputs rather than of a proxy for them.
//
// Deliberately not an AnnotationFrame compared with sameAnnotationFrame(): that
// comparison is voxel-size sensitive, so it would differ for two volumes of one
// scan whose recorded micrometre figures disagree and force a rebuild before
// umbilicusFactor could show that nothing had moved. Whatever the voxel size
// contributed, it contributed through the factor.
//
// The scale *source* is deliberately absent: identical resolved geometry must not
// rebuild because its provenance changed.
struct OrientationKey {
    // The annotation frame the views were built in. Compared through
    // sameAnnotationGrid/sameAnnotationFrame below rather than field by field,
    // because a differing physical voxel size does not always matter.
    AnnotationFrame frame;
    // The session volume's own shape. The volume-centre fallback and the legacy
    // reading both use it directly, so two volumes sharing an annotation grid can
    // still need different orientation.
    std::array<int, 3> rawVolumeShapeXyz{0, 0, 0};
    UmbilicusOrientationMode mode = UmbilicusOrientationMode::VolumeCentre;
    // Which reading produced the scale, when one was applied. This is what decides
    // whether a changed physical voxel size reached the geometry: through stamped
    // dimensions or grid inference the µm figure played no part.
    //
    // Deliberately the *recorded* source and not a live factor. A factor can only
    // be recomputed by re-resolving the umbilicus, which is filesystem work, and a
    // comparison that reads the cached factor would compare the old value against
    // itself and conclude nothing had changed — missing the one case it exists for.
    std::optional<vc::core::util::UmbilicusScaleSource> scaleSource;
    // Legacy reading only: the registration transform it went through, if any.
    std::string transformPath;
    std::uintmax_t transformSize = 0;
};

inline bool sameOrientationKey(const OrientationKey& a, const OrientationKey& b)
{
    if (a.rawVolumeShapeXyz != b.rawVolumeShapeXyz)
        return false;
    if (a.mode != b.mode)
        return false;
    if (a.transformPath != b.transformPath || a.transformSize != b.transformSize)
        return false;
    // Different voxel counts: the geometry is about another grid.
    if (!sameAnnotationGrid(a.frame, b.frame))
        return false;
    if (sameAnnotationFrame(a.frame, b.frame))
        return true;
    // Same counts, different physical scale. It only reached the geometry if the
    // scale was derived from it; through stamped dimensions or grid inference the
    // views are unaffected and rebuilding them would change nothing.
    return a.scaleSource != vc::core::util::UmbilicusScaleSource::StampedVoxelSize &&
           b.scaleSource != vc::core::util::UmbilicusScaleSource::StampedVoxelSize;
}

} // namespace vc3d::annotation
