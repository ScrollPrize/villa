#pragma once

// Binds the skeleton backend to a stored fiber prediction volume: builds the
// tube crop and serves corner blocks and grid-aligned samples through the
// existing prediction and normal samplers.

#include "vc/fiber_tracer/FiberSkeletonFields.hpp"
#include "vc/fiber_tracer/FiberTrace.hpp"

#include <memory>
#include <vector>

namespace vc::fiber_tracer
{

// Half-open prediction-grid box covering a tube of `tubeRadiusBase` around
// every polyline, padded by `cropMarginBase` and clipped to the stored grid.
// `presence` is filled from the stored grid and zeroed outside the tube.
[[nodiscard]] FiberSkeletonCrop buildFiberSkeletonCrop(
    const FiberPredictionField& predictions,
    const std::vector<std::vector<cv::Vec3d>>& tubePolylinesBaseXYZ,
    double tubeRadiusBase, double cropMarginBase, int parallelThreads);

// Field provider over a stored prediction volume. Direction and normal values
// are fetched in batches through the existing samplers and memoized by grid
// index for the lifetime of the provider.
class FiberSkeletonStoredFields final : public FiberSkeletonFieldProvider
{
public:
    FiberSkeletonStoredFields(
        const FiberPredictionField& predictions,
        const vc::lasagna::NormalSampler& normalSampler,
        double normalWorkingToBaseScale, FiberSkeletonCrop crop,
        int parallelThreads);
    ~FiberSkeletonStoredFields() override;

    void loadCorners(
        const std::vector<cv::Vec3d>& pointsBaseXYZ,
        const std::vector<cv::Vec3d>& referenceDirections,
        std::vector<FiberSkeletonCornerBlock>& blocks) override;
    void loadVoxels(
        const std::vector<std::size_t>& flatCropIndices,
        std::vector<FiberSkeletonFieldSample>& samples) override;

    [[nodiscard]] const FiberSkeletonCrop& crop() const noexcept override;
    [[nodiscard]] double predictionToBaseScale() const noexcept override;

    [[nodiscard]] std::size_t cachedVoxels() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vc::fiber_tracer
