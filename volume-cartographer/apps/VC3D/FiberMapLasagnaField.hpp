#pragma once

#include "FiberMapBentRays.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>

namespace vc::lasagna {
class LasagnaDataset;
class LasagnaNormalSampler;
}

namespace vc3d::fiber_map::bent
{

// The sheet-normal field of a Lasagna dataset: its nx/ny compact axis
// tensor decoded to the unoriented 3D axis of the predicted sheet normal
// (the stored axis has a non-negative z component - an axis, never an
// orientation). The dataset's workingToBaseScale must map the FIBERS' frame
// (the annotation frame) to the lasagna base; the caller resolves it from
// the annotation frame and the manifest's base shape before constructing
// the sampler.
//
// Identity: the manifest location, the digest of the manifest text, the
// digest of every channel group's complete array metadata as opened (every
// field of the v2 or v3 metadata: shape, chunks, dtype, fill value, byte
// order, compressor and level, quantiser, dimension separator, filters,
// codec pipeline, chunk key encoding, sharding, node type) in manifest
// order, and the scale. The contract behind it: a channel location is immutable
// data while VC3D runs (the open-data locations carry their version in the
// path; a local dataset edited in place requires a restart). The identity
// deliberately carries no modification time or remote object token: none
// exists for remote stores, and none would be honoured by the sampler's
// path-keyed process caches, so pretending to see in-place edits would
// break fresh-equals-cached rather than keep it.
class LasagnaSheetNormalField final : public SheetNormalField {
public:
    LasagnaSheetNormalField(std::shared_ptr<const vc::lasagna::LasagnaNormalSampler> sampler,
                            std::string identity, int threads);
    ~LasagnaSheetNormalField() override;

    [[nodiscard]] std::optional<cv::Vec3d> axis(const cv::Vec3d& volumePoint) const override;
    void axes(const std::vector<cv::Vec3d>& points,
              std::vector<std::optional<cv::Vec3d>>& out) const override;
    [[nodiscard]] std::string identity() const override { return identity_; }

private:
    std::shared_ptr<const vc::lasagna::LasagnaNormalSampler> sampler_;
    std::string identity_;
    int threads_;
};

// The identity string of a dataset (see above), for a dataset whose
// manifest has been opened; `workingToBaseScale` is the scale the sampler
// will be built with.
[[nodiscard]] std::string lasagnaFieldIdentity(const vc::lasagna::LasagnaDataset& dataset,
                                               double workingToBaseScale);

// The scale that carries the fibers' frame (the annotation frame, whose
// extent in voxels is `annotationExtentXyz`) into a lasagna manifest's base
// grid (`baseShapeZYX`), the dyadic ratio of the two grids - exactly as the
// tracer resolves its normal field (annotation at L0 against a base at L1
// gives 0.5). 1 when the manifest states no base shape; throws when it does
// and the annotation extent is unknown (no scale can be resolved), or when
// the grids are not a dyadic pair.
[[nodiscard]] double sheetFieldWorkingToBaseScale(
    const std::array<double, 3>& annotationExtentXyz,
    const std::optional<std::array<std::size_t, 3>>& baseShapeZYX);

} // namespace vc3d::fiber_map::bent
