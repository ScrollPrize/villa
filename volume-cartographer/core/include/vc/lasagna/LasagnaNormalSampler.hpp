#pragma once

#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/ChannelSampler.hpp"
#include "vc/lasagna/LineModel.hpp"

#include <cstddef>
#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>

namespace vc::lasagna
{

void validateLasagnaNormalDatasetStructure(const LasagnaDataset& dataset);

struct LasagnaNormalSamplerOptions {
    size_t maxCachedBytes = 512ULL * 1024ULL * 1024ULL;
};

class LasagnaNormalSampler final : public NormalSampler
{
public:
    struct FloatNormalSample {
        cv::Vec3f normal{0.0f, 0.0f, 0.0f};
        bool valid = false;
    };

    explicit LasagnaNormalSampler(const LasagnaDataset& dataset, LasagnaNormalSamplerOptions options = {});
    ~LasagnaNormalSampler() override;

    LasagnaNormalSampler(const LasagnaNormalSampler&) = delete;
    LasagnaNormalSampler& operator=(const LasagnaNormalSampler&) = delete;
    LasagnaNormalSampler(LasagnaNormalSampler&&) noexcept;
    LasagnaNormalSampler& operator=(LasagnaNormalSampler&&) noexcept;

    [[nodiscard]] bool supportsConcurrentSampling() const noexcept override { return true; }
    [[nodiscard]] NormalSample sampleNormal(const cv::Vec3d& volumePoint) const override;
    [[nodiscard]] std::optional<double> sampleWindingDensity(const cv::Vec3d& volumePoint) const;
    [[nodiscard]] std::optional<double> samplePredDt(const cv::Vec3d& volumePoint) const;
    [[nodiscard]] bool hasPredDtChannel() const;
    [[nodiscard]] std::optional<double> predDtSpacing() const;
    [[nodiscard]] double windingDistance(const cv::Vec3d& a, const cv::Vec3d& b, double stepVx = 8.0) const;
    [[nodiscard]] double normalAlignedWindingDistance(
        const cv::Vec3d& a,
        const cv::Vec3d& b,
        double stepVx = 8.0) const;
    [[nodiscard]] NormalSampleWithDerivative sampleNormalWithDerivative(const cv::Vec3d& volumePoint) const override;
    [[nodiscard]] NormalPrefetchReport prefetchNormalSamples(const std::vector<cv::Vec3d>& volumePoints, bool withDerivative) const override;
    [[nodiscard]] NormalBatchReport sampleNormalBatch(
        const std::vector<cv::Vec3d>& volumePoints, bool withDerivative, std::vector<NormalSampleWithDerivative>& samples) const override;
    [[nodiscard]] NormalBatchReport sampleNormalBatch(
        const std::vector<cv::Vec3d>& volumePoints, bool withDerivative, int parallelThreads, std::vector<NormalSampleWithDerivative>& samples) const override;
    [[nodiscard]] NormalBatchReport sampleNormalBatch(const std::vector<cv::Vec3f>& volumePoints, int parallelThreads, std::vector<FloatNormalSample>& samples) const;
    [[nodiscard]] std::array<const LasagnaChannelCornerSampler*, 3> groupedCornerSamplers() const noexcept;
    void materializeGroupedCorners(
        const std::vector<std::vector<LasagnaCornerSample>>& corners, size_t firstVolume, int parallelThreads, std::vector<FloatNormalSample>& samples) const;
    void materializeGroupedCorners(const LasagnaCornerBatch& corners, size_t firstVolume, int parallelThreads, std::vector<FloatNormalSample>& samples) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vc::lasagna
