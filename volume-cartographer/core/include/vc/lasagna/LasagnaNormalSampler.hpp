#pragma once

#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LineModel.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>

namespace vc::lasagna {

struct LasagnaNormalSamplerOptions {
    size_t maxCachedChunks = 128;
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

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vc::lasagna
