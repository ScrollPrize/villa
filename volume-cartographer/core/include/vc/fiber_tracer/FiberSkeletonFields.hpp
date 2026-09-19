#pragma once

// Differentiable access to the prediction fields for the skeleton backend.
//
// A solver step needs presence, fiber direction and the Lasagna normal at
// moving polyline vertices. Rather than resampling inside the residual, the
// eight grid corners around each vertex are batch-loaded once per solver
// iteration; the residual then interpolates those constants trilinearly, so
// Ceres Jets differentiate through the interpolation weights.

#include "vc/fiber_tracer/FiberSkeleton.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer
{

// The eight corners of one grid cell. Corner order is
// `corner = dz * 4 + dy * 2 + dx` with each offset in {0, 1}.
struct FiberSkeletonCornerBlock {
    cv::Vec3d cellOriginBaseXYZ{0.0, 0.0, 0.0};
    double spacingBase = 1.0;
    std::array<double, 8> presence{};
    // Sign-aligned to `referenceDirection` when the block was loaded, so the
    // ambiguous prediction sign is a constant of the current evaluation.
    std::array<cv::Vec3d, 8> direction{};
    std::array<cv::Vec3d, 8> normal{};
    std::array<std::uint8_t, 8> valid{};
    std::array<std::uint8_t, 8> normalValid{};
    cv::Vec3d referenceDirection{0.0, 0.0, 0.0};
    bool inside = false;   // the whole cell lies inside the loaded crop
};

// Scalar convenience sample, used by port optimization and the discrete
// initial loss. Interpolates the same way the residuals do.
struct FiberSkeletonFieldSample {
    double presence = 0.0;
    cv::Vec3d direction{0.0, 0.0, 0.0};
    cv::Vec3d normal{0.0, 0.0, 0.0};
    bool valid = false;
    bool normalValid = false;
};

// Batch corner loader. Implementations are not required to be thread safe.
class FiberSkeletonFieldProvider
{
public:
    virtual ~FiberSkeletonFieldProvider() = default;

    // Loads one block per query point. `referenceDirections` fixes the sign of
    // the ambiguous predicted direction for this evaluation and must have the
    // same size as `pointsBaseXYZ`. `blocks` is resized to match.
    virtual void loadCorners(
        const std::vector<cv::Vec3d>& pointsBaseXYZ,
        const std::vector<cv::Vec3d>& referenceDirections,
        std::vector<FiberSkeletonCornerBlock>& blocks) = 0;

    // Grid-aligned samples at whole crop voxels, addressed by flat crop
    // index. This is the path used by port optimization and by the discrete
    // initial loss along skeleton voxels, where no interpolation is wanted.
    virtual void loadVoxels(
        const std::vector<std::size_t>& flatCropIndices,
        std::vector<FiberSkeletonFieldSample>& samples) = 0;

    [[nodiscard]] virtual const FiberSkeletonCrop& crop() const noexcept = 0;
    [[nodiscard]] virtual double predictionToBaseScale() const noexcept = 0;
};

[[nodiscard]] FiberSkeletonFieldSample interpolateFiberSkeletonBlock(
    const FiberSkeletonCornerBlock& block, const cv::Vec3d& pointBaseXYZ);

// Batch scalar sampling through a provider.
void sampleFiberSkeletonFields(
    FiberSkeletonFieldProvider& provider,
    const std::vector<cv::Vec3d>& pointsBaseXYZ,
    const std::vector<cv::Vec3d>& referenceDirections,
    std::vector<FiberSkeletonFieldSample>& samples);

}  // namespace vc::fiber_tracer
