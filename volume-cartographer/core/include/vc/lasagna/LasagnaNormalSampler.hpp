#pragma once

#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LineModel.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>

namespace vc::lasagna {

struct LasagnaNormalSamplerOptions {
    size_t maxCachedBytes = 512ULL * 1024ULL * 1024ULL;
};

class LasagnaNormalSampler final : public NormalSampler {
public:
    explicit LasagnaNormalSampler(
        const LasagnaDataset& dataset,
        LasagnaNormalSamplerOptions options = {});
    ~LasagnaNormalSampler() override;

    LasagnaNormalSampler(const LasagnaNormalSampler&) = delete;
    LasagnaNormalSampler& operator=(const LasagnaNormalSampler&) = delete;
    LasagnaNormalSampler(LasagnaNormalSampler&&) noexcept;
    LasagnaNormalSampler& operator=(LasagnaNormalSampler&&) noexcept;

    [[nodiscard]] NormalSample sampleNormal(const cv::Vec3d& volumePoint) const override;
    [[nodiscard]] std::optional<double> sampleWindingDensity(const cv::Vec3d& volumePoint) const;
    [[nodiscard]] double windingDistance(const cv::Vec3d& a,
                                         const cv::Vec3d& b,
                                         double stepVx = 8.0) const;
    [[nodiscard]] NormalSampleWithDerivative sampleNormalWithDerivative(
        const cv::Vec3d& volumePoint) const override;
    [[nodiscard]] NormalPrefetchReport prefetchNormalSamples(
        const std::vector<cv::Vec3d>& volumePoints,
        bool withDerivative) const override;
    [[nodiscard]] NormalBatchReport sampleNormalBatch(
        const std::vector<cv::Vec3d>& volumePoints,
        bool withDerivative,
        std::vector<NormalSampleWithDerivative>& samples) const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vc::lasagna
